# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
from typing import Any, Dict, List, Optional
from packaging import version

from pymongo import MongoClient, TEXT
from pymongo.operations import SearchIndexModel, UpdateOne
import certifi
from numpy.typing import NDArray

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO

from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
    RERANKER_TYPE_WEIGHTED,
    ChunkForDeletion,
)
  
from .config import MongoDBIOConfig, SearchMode, PipelineConfig

log = logging.getLogger(__name__)
CHUNK_ID_KEY = "_chunk_id"


class MongoDBAtlasIndex(EmbeddingIndex):

    def __init__(self, client: MongoClient, namespace: str, embeddings_key: str, 
                 embedding_dimension: int, index_name: str,
                 config: MongoDBIOConfig):
        self.client = client
        self.namespace = namespace
        self.embeddings_key = embeddings_key
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.config = config
        self._check_mongodb_version()

    def _check_mongodb_version(self):
        """Check MongoDB version for feature compatibility"""
        try:
            server_info = self.client.server_info()
            self.server_version = server_info.get('version', '8.0')
            self.supports_rank_fusion = version.parse(self.server_version) >= version.parse('8.1')
            
            # Check if we're on Atlas (for Atlas Search features)
            self.is_atlas = 'mongodb.net' in self.config.connection_str or 'mongodb-dev.net' in self.config.connection_str
            
            log.info("MongoDB version: %s", self.server_version)
            log.info("Native $rankFusion: %s", self.supports_rank_fusion)
            log.info("Atlas deployment: %s", self.is_atlas)
        except Exception as e:
            log.warning("Could not determine MongoDB version: %s", e)
            self.server_version = '8.0'
            self.supports_rank_fusion = False
            self.is_atlas = False

    def _get_index_config(self, collection, index_name):
        """Get existing index configuration"""
        try:
            idxs = list(collection.list_search_indexes())
            for ele in idxs:
                if ele["name"] == index_name:
                    return ele
        except Exception as e:
            log.warning("Error listing search indexes: %s", e)
        return None

    def _create_vector_search_index(self):
        """Create vector search index for Atlas"""
        return SearchIndexModel(
            name=self.index_name,
            type="vectorSearch",
            definition={
                "fields": [{
                    "path": self.embeddings_key,
                    "type": "vector",
                    "numDimensions": self.embedding_dimension,
                    "similarity": "cosine"
                }]
            }
        )

    def _create_text_search_index(self):
        """Create text search index for Atlas Search"""
        # Dynamic mapping for flexibility
        definition = {
            "mappings": {
                "dynamic": True,
                "fields": {}
            }
        }
        
        # Add specific field mappings if configured
        for field in self.config.text_search_fields:
            definition["mappings"]["fields"][field] = {
                "type": "string",
                "analyzer": "lucene.standard"
            }
        
        return SearchIndexModel(
            name=self.config.text_index_name,
            type="search",
            definition=definition
        )

    def _check_n_create_indices(self):
        """Create necessary search indices"""
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        
        # Create vector search index
        if not self._get_index_config(collection, self.index_name):
            log.info("Creating vector search index: %s", self.index_name)
            try:
                vector_index = self._create_vector_search_index()
                collection.create_search_index(vector_index)
                log.info("Vector search index created successfully")
            except Exception as e:
                log.error("Failed to create vector search index: %s", e)
        
        # Create text search index if needed
        if (self.config.search_mode in [SearchMode.FULL_TEXT, SearchMode.HYBRID, 
                                       SearchMode.HYBRID_GRAPH, SearchMode.NATIVE_RANK_FUSION] and
            self.config.text_index_name and 
            self.config.text_index_name != self.index_name):
            
            if not self._get_index_config(collection, self.config.text_index_name):
                log.info("Creating text search index: %s", self.config.text_index_name)
                try:
                    text_index = self._create_text_search_index()
                    collection.create_search_index(text_index)
                    log.info("Text search index created successfully")
                except Exception as e:
                    log.error("Failed to create text search index: %s", e)
        
        # Create standard text index as fallback
        try:
            text_indices = [(field, TEXT) for field in self.config.text_search_fields]
            collection.create_index(text_indices, name="fallback_text_index")
            log.info("Created fallback text index")
        except Exception as e:
            if "already exists" not in str(e):
                log.warning("Could not create fallback text index: %s", e)

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        """Add chunks with embeddings and metadata"""
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        operations = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Prefer existing chunk_id if present
            chunk_id = (
                getattr(chunk, "chunk_id", None)
                or chunk.metadata.get("chunk_id")
                or f"chunk-{i}"
            )
            
            # Extract text content for text search
            text_content = ""
            if hasattr(chunk.content, 'text'):
                text_content = chunk.content.text
            elif isinstance(chunk.content, str):
                text_content = chunk.content
            elif isinstance(chunk.content, list):
                text_content = " ".join([str(c) for c in chunk.content if c])
            
            # Prepare document with all necessary fields
            doc_data = {
                CHUNK_ID_KEY: chunk_id,
                "chunk_content": json.loads(chunk.model_dump_json()),
                self.embeddings_key: embedding.tolist(),
                "text": text_content,
                "title": chunk.metadata.get('title', ''),
                "metadata": chunk.metadata,
            }
            
            # Add graph relationship fields if configured
            if self.config.graph_lookup_config:
                # Add related document IDs for graph traversal
                if 'related_ids' in chunk.metadata:
                    doc_data['metadata']['related_ids'] = chunk.metadata['related_ids']
                if 'parent_id' in chunk.metadata:
                    doc_data['metadata']['parent_id'] = chunk.metadata['parent_id']
            
            operations.append(
                UpdateOne(
                    {CHUNK_ID_KEY: chunk_id},
                    {"$set": doc_data},
                    upsert=True,
                )
            )
        
        # Perform bulk write
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        result = collection.bulk_write(operations)
        log.info(
            "Bulk write results - Inserted: %s, Modified: %s",
            result.inserted_count,
            result.modified_count,
        )
        
        # Create indices
        self._check_n_create_indices()

    def _build_native_rank_fusion_pipeline(self, embedding: NDArray, query_text: str, 
                                          k: int, filters: Optional[Dict] = None) -> List[Dict]:
        """Build native MongoDB 8.1+ $rankFusion pipeline"""
        pipelines = {}
        weights = {}
        
        # Build pipelines from configuration
        for pipeline_cfg in self.config.rank_fusion_pipelines:
            pipeline_stages = []
            
            if pipeline_cfg.type == "vectorSearch":
                vector_stage = {
                    "$vectorSearch": {
                        "index": pipeline_cfg.config.get("index", self.index_name),
                        "path": self.embeddings_key,
                        "queryVector": embedding.tolist(),
                        "numCandidates": pipeline_cfg.config.get("numCandidates", 100),
                        "limit": pipeline_cfg.limit
                    }
                }
                # Add filters if provided
                if filters:
                    vector_stage["$vectorSearch"]["filter"] = filters
                pipeline_stages.append(vector_stage)
                
            elif pipeline_cfg.type == "search" and query_text:
                search_stage = {
                    "$search": {
                        "index": pipeline_cfg.config.get("index", self.config.text_index_name),
                    }
                }
                
                # Configure search operator
                operator_type = pipeline_cfg.config.get("operator", "text")
                if operator_type == "phrase":
                    search_stage["$search"]["phrase"] = {
                        "query": query_text,
                        "path": self.config.text_search_fields
                    }
                elif operator_type == "compound":
                    search_stage["$search"]["compound"] = {
                        "must": [{
                            "text": {
                                "query": query_text,
                                "path": self.config.text_search_fields
                            }
                        }]
                    }
                else:  # default text operator
                    search_stage["$search"]["text"] = {
                        "query": query_text,
                        "path": self.config.text_search_fields
                    }
                
                pipeline_stages.append(search_stage)
                pipeline_stages.append({"$limit": pipeline_cfg.limit})
                
            elif pipeline_cfg.type == "match":
                # Standard MongoDB query
                match_query = pipeline_cfg.config.get("query", {})
                if query_text and not match_query:
                    # Default to text search if no query provided
                    match_query = {"$text": {"$search": query_text}}
                pipeline_stages.append({"$match": match_query})
                pipeline_stages.append({"$sort": pipeline_cfg.config.get("sort", {"score": {"$meta": "textScore"}})})
                pipeline_stages.append({"$limit": pipeline_cfg.limit})
                
            elif pipeline_cfg.type == "geoNear":
                geo_config = pipeline_cfg.config
                if "near" in geo_config:
                    pipeline_stages.append({
                        "$geoNear": {
                            "near": geo_config["near"],
                            "key": geo_config.get("key", "location"),
                            "spherical": geo_config.get("spherical", True),
                            "maxDistance": geo_config.get("maxDistance", 10000),
                            "query": filters or {},
                            "limit": pipeline_cfg.limit
                        }
                    })
            
            if pipeline_stages:
                pipelines[pipeline_cfg.name] = pipeline_stages
                weights[pipeline_cfg.name] = pipeline_cfg.weight
        
        # Build the rank fusion pipeline
        rank_fusion_pipeline = [
            {
                "$rankFusion": {
                    "input": {"pipelines": pipelines},
                    "combination": {"weights": weights} if weights else {},
                    "scoreDetails": self.config.enable_score_details
                }
            },
            {"$limit": k}
        ]
        
        # Add score details if enabled
        if self.config.enable_score_details:
            rank_fusion_pipeline.append({
                "$addFields": {
                    "scoreDetails": {"$meta": "scoreDetails"},
                    "score": {"$meta": "score"}
                }
            })
        else:
            rank_fusion_pipeline.append({
                "$addFields": {
                    "score": {"$meta": "score"}
                }
            })
        
        # Add graph lookup enhancement if configured
        if self.config.enable_graph_enhancement and self.config.graph_lookup_config:
            rank_fusion_pipeline.extend(self._build_graph_lookup_stages())
        
        # Final projection
        rank_fusion_pipeline.append(
            {
                "$project": {
                    CHUNK_ID_KEY: 1,
                    "chunk_content": 1,
                    "score": 1,
                    "scoreDetails": 1 if self.config.enable_score_details else 0,
                    "graph_connections": 1 if self.config.enable_graph_enhancement else 0,
                    "_id": 0,
                }
            }
        )
        
        return rank_fusion_pipeline


    def _build_graph_lookup_stages(self) -> List[Dict]:
        """Build $graphLookup stages for relationship traversal"""
        if not self.config.graph_lookup_config:
            return []
        
        cfg = self.config.graph_lookup_config
        _, collection_name = self.namespace.split(".")
        
        # Now cfg is properly typed as GraphLookupConfig
        graph_lookup = {
            "$graphLookup": {
                "from": cfg.from_collection or collection_name,
                "startWith": cfg.start_with,
                "connectFromField": cfg.connect_from_field,
                "connectToField": cfg.connect_to_field,
                "as": cfg.as_field,
                "maxDepth": cfg.max_depth,
            }
        }
        
        # Add optional fields
        if cfg.depth_field:
            graph_lookup["$graphLookup"]["depthField"] = cfg.depth_field
        
        if cfg.restrict_search_with_match:
            graph_lookup["$graphLookup"]["restrictSearchWithMatch"] = cfg.restrict_search_with_match
        
        return [graph_lookup]

    async def _enhance_with_graph_lookup(self, initial_results: List[Dict], k: int) -> List[Dict]:
        """Enhance results with graph traversal"""
        if not self.config.graph_lookup_config or not initial_results:
            return initial_results
        
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        cfg = self.config.graph_lookup_config
        
        # Extract starting values for graph traversal
        start_values = []
        for result in initial_results[: k // 2]:
            if "chunk_content" in result and "metadata" in result["chunk_content"]:
                metadata = result["chunk_content"]["metadata"]
                
                # Extract the field specified in start_with (removing the $ prefix)
                start_field = cfg.start_with.lstrip("$").replace("metadata.", "")
                
                if "document_id" in metadata:
                    start_values.append(metadata["document_id"])
                if start_field in metadata:
                    value = metadata[start_field]
                    if isinstance(value, list):
                        start_values.extend(value)
                    else:
                        start_values.append(value)
        
        if not start_values:
            return initial_results
        
        # Build graph lookup pipeline with proper GraphLookupConfig
        graph_pipeline = [
            {"$match": {cfg.connect_to_field: {"$in": list(set(start_values))}}},
            {
                "$graphLookup": {
                    "from": cfg.from_collection or collection_name,
                    "startWith": cfg.start_with,
                    "connectFromField": cfg.connect_from_field,
                    "connectToField": cfg.connect_to_field,
                    "as": cfg.as_field,
                    "maxDepth": cfg.max_depth,
                    "depthField": cfg.depth_field if cfg.depth_field else "depth",
                }
            },
            {"$unwind": f"${cfg.as_field}"},
            {"$replaceRoot": {"newRoot": f"${cfg.as_field}"}},
            {
                "$project": {
                    CHUNK_ID_KEY: 1,
                    "chunk_content": 1,
                    (cfg.depth_field or "depth"): 1,
                    "score": {"$divide": [1, {"$add": [f"${cfg.depth_field or 'depth'}", 1]}]},
                    "_id": 0,
                }
            },
            {"$limit": k // 2},
        ]
        
        # Add restriction if specified
        if cfg.restrict_search_with_match:
            graph_pipeline[1]["$graphLookup"]["restrictSearchWithMatch"] = cfg.restrict_search_with_match
        
        try:
            graph_results = list(collection.aggregate(graph_pipeline, allowDiskUse=True))
            
            # Combine with original results using RRF
            combined = self._reciprocal_rank_fusion([initial_results, graph_results], [0.7, 0.3], k)
            return combined
        except Exception as e:
            log.warning("Graph lookup enhancement failed: %s", e)
            return initial_results

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        """Vector-only search using $vectorSearch."""
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": self.embeddings_key,
                        "queryVector": embedding.tolist(),
                        "numCandidates": min(k * 10, 1000),
                        "limit": k,
                    }
                },
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$match": {"score": {"$gte": score_threshold}}},
                {"$project": {CHUNK_ID_KEY: 1, "chunk_content": 1, "score": 1, "_id": 0}},
            ]
            results = list(collection.aggregate(pipeline))
        except Exception as e:
            log.error("Vector query failed: %s", e)
            results = []
        return self._format_results(results)

    async def query_keyword(self, query_string: str, k: int, score_threshold: float) -> QueryChunksResponse:
        """Text-only search using Atlas $search or fallback to $text."""
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        results: List[Dict] = []
        # Prefer Atlas Search if available
        pipeline = [
            {
                "$search": {
                    "index": self.config.text_index_name,
                    "text": {"query": query_string, "path": self.config.text_search_fields},
                }
            },
            {"$addFields": {"score": {"$meta": "searchScore"}}},
            {"$match": {"score": {"$gte": score_threshold}}},
            {"$limit": k},
            {"$project": {CHUNK_ID_KEY: 1, "chunk_content": 1, "score": 1, "_id": 0}},
        ]
        try:
            results = list(collection.aggregate(pipeline))
        except Exception as e:
            log.info("Atlas $search failed, falling back to $text: %s", e)
            try:
                # Ensure a text index exists on configured fields; _check_n_create_indices tries to create one
                cursor = (
                    collection.find(
                        {"$text": {"$search": query_string}},
                        {CHUNK_ID_KEY: 1, "chunk_content": 1, "score": {"$meta": "textScore"}, "_id": 0},
                    )
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(k)
                )
                results = [doc for doc in cursor if float(doc.get("score", 0.0)) >= score_threshold]
            except Exception as ee:
                log.error("Text query fallback failed: %s", ee)
                results = []
        return self._format_results(results)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """Hybrid search combining vector and keyword results.

        Uses native $rankFusion on MongoDB >= 8.1 if pipelines are configured,
        otherwise performs manual fusion (RRF or Weighted).
        """
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]

        # Try native $rankFusion when available and configured
        if self.supports_rank_fusion and self.config.rank_fusion_pipelines:
            try:
                pipeline = self._build_native_rank_fusion_pipeline(embedding, query_string, k)
                native_results = list(collection.aggregate(pipeline, allowDiskUse=True))
                # Optionally enhance with graph traversal
                if self.config.enable_graph_enhancement and self.config.graph_lookup_config:
                    native_results = await self._enhance_with_graph_lookup(native_results, k)
                return self._format_results(native_results)
            except Exception as e:
                log.warning("Native $rankFusion failed, falling back to manual fusion: %s", e)

        # Manual hybrid: run vector and text searches and fuse
        vector_results: List[Dict] = []
        text_results: List[Dict] = []

        # Vector part
        try:
            vs_pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": self.embeddings_key,
                        "queryVector": embedding.tolist(),
                        "numCandidates": min(k * 10, 1000),
                        "limit": k,
                    }
                },
                {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
                {"$project": {CHUNK_ID_KEY: 1, "chunk_content": 1, "score": 1, "_id": 0}},
            ]
            vector_results = list(collection.aggregate(vs_pipeline))
        except Exception as e:
            log.warning("Vector search part failed: %s", e)

        # Text part
        try:
            text_pipeline = [
                {
                    "$search": {
                        "index": self.config.text_index_name,
                        "text": {"query": query_string, "path": self.config.text_search_fields},
                    }
                },
                {"$addFields": {"score": {"$meta": "searchScore"}}},
                {"$project": {CHUNK_ID_KEY: 1, "chunk_content": 1, "score": 1, "_id": 0}},
                {"$limit": k},
            ]
            text_results = list(collection.aggregate(text_pipeline))
        except Exception:
            # Fallback to $text
            try:
                cursor = (
                    collection.find(
                        {"$text": {"$search": query_string}},
                        {CHUNK_ID_KEY: 1, "chunk_content": 1, "score": {"$meta": "textScore"}, "_id": 0},
                    )
                    .sort([("score", {"$meta": "textScore"})])
                    .limit(k)
                )
                text_results = list(cursor)
            except Exception as e:
                log.warning("Text search part failed: %s", e)

        # Apply score threshold pre-filtering
        vector_results = [r for r in vector_results if float(r.get("score", 0.0)) >= score_threshold]
        text_results = [r for r in text_results if float(r.get("score", 0.0)) >= score_threshold]

        # Rerank/fuse
        if reranker_type == RERANKER_TYPE_WEIGHTED:
            alpha = (reranker_params or {}).get("alpha", 0.5)
            fused = self._weighted_fusion([vector_results, text_results], [alpha, 1 - alpha], k)
        else:
            # impact_factor is not used directly in current RRF helper
            # Our RRF helper uses fixed 60 in denominator; approximate by scaling ranks is non-trivial.
            # We'll just use standard RRF with weights = 1 for both lists.
            fused = self._reciprocal_rank_fusion([vector_results, text_results], [1.0, 1.0], k)

        # Optionally enhance with graph traversal
        if self.config.enable_graph_enhancement and self.config.graph_lookup_config:
            fused = await self._enhance_with_graph_lookup(fused, k)

        return self._format_results(fused)

    def _weighted_fusion(self, result_sets: List[List[Dict]], weights: List[float], k: int) -> List[Dict]:
        """Simple weighted-score fusion by normalizing ranks then weighting."""
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict] = {}
        for results, w in zip(result_sets, weights):
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get(CHUNK_ID_KEY)
                if not doc_id:
                    continue
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc
                    doc_scores[doc_id] = 0.0
                # Use reciprocal rank as base, then weight
                doc_scores[doc_id] += w * (1.0 / (60 + rank))
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        out: List[Dict] = []
        for doc_id, score in sorted_docs[:k]:
            d = doc_data[doc_id].copy()
            d["score"] = score
            out.append(d)
        return out

    def _reciprocal_rank_fusion(self, result_sets: List[List[Dict]], weights: List[float], k: int) -> List[Dict]:
        """Implement reciprocal rank fusion with weights"""
        doc_scores = {}
        doc_data = {}
        
        for result_set, weight in zip(result_sets, weights):
            for rank, doc in enumerate(result_set, 1):
                doc_id = doc.get(CHUNK_ID_KEY)
                if doc_id:
                    if doc_id not in doc_data:
                        doc_data[doc_id] = doc
                        doc_scores[doc_id] = 0
                    # RRF formula: weight * (1 / (60 + rank))
                    doc_scores[doc_id] += weight * (1.0 / (60 + rank))
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_docs[:k]:
            doc = doc_data[doc_id].copy()
            doc['score'] = score
            results.append(doc)
        
        return results

    def _format_results(self, results: List[Dict]) -> QueryChunksResponse:
        """Format results into QueryChunksResponse"""
        chunks = []
        scores = []
        
        for result in results:
            if 'chunk_content' in result:
                content = result['chunk_content']
                chunk = Chunk(
                    metadata=content.get('metadata', {}),
                    content=content.get('content', '')
                )
                chunks.append(chunk)
                scores.append(float(result.get('score', 0.0)))
        
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: List[ChunkForDeletion]) -> None:
        """Delete chunks by chunk_id from the MongoDB collection."""
        if not chunks_for_deletion:
            return
        db, collection_name = self.namespace.split(".")
        collection = self.client[db][collection_name]
        try:
            ids = [c.chunk_id for c in chunks_for_deletion]
            collection.delete_many({CHUNK_ID_KEY: {"$in": ids}})
        except Exception as e:
            log.error("Failed to delete chunks: %s", e)

    async def delete(self):
        """Delete the database"""
        db, _ = self.namespace.split(".")
        self.client.drop_database(db)


class MongoDBIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(self, config: MongoDBIOConfig, inference_api: Api.inference):
        self.config = config
        self.inference_api = inference_api
        self.cache = {}
        self.client: MongoClient | None = None
        self.vector_db_store = None

    async def initialize(self) -> None:
        """Initialize MongoDB connection and auto-configure for version"""
        self.client = MongoClient(
            self.config.connection_str,
            tlsCAFile=certifi.where(),
        )
        
        # Auto-detect MongoDB version and capabilities
        try: 
            server_info = self.client.server_info()
            version_str = server_info.get('version', 'unknown')
            log.info("Connected to MongoDB %s", version_str)
            
            # Auto-upgrade to native features if available
            if version.parse(version_str) >= version.parse('8.1'):
                log.info("MongoDB 8.1+ detected - native $rankFusion available")
                
                # Auto-configure rank fusion pipelines if not set
                if (self.config.search_mode in [SearchMode.HYBRID, SearchMode.HYBRID_GRAPH] and 
                    not self.config.rank_fusion_pipelines):
                    
                    log.info("Auto-configuring native rank fusion pipelines")
                    self.config.search_mode = SearchMode.NATIVE_RANK_FUSION
                    self.config.rank_fusion_pipelines = [
                        PipelineConfig(
                            name="vector_search",
                            type="vectorSearch",
                            weight=1.0,
                            limit=20,
                            config={"numCandidates": 100}
                        ),
                        PipelineConfig(
                            name="text_search",
                            type="search",
                            weight=1.0,
                            limit=20,
                            config={"operator": "text"}
                        )
                    ]
        except Exception as e:
            log.warning("Could not determine MongoDB version: %s", e)

    async def shutdown(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        """Register a vector database"""
        index = MongoDBAtlasIndex(
            client=self.client,
            namespace=self.config.namespace,
            embeddings_key=self.config.embeddings_key,
            embedding_dimension=vector_db.embedding_dimension,
            index_name=self.config.index_name,
            config=self.config,
        )
        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db=vector_db,
            index=index,
            inference_api=self.inference_api,
        )

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex:
        """Get or create cached vector DB index"""
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]
        
        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        self.cache[vector_db_id] = VectorDBWithIndex(
            vector_db=vector_db,
            index=MongoDBAtlasIndex(
                client=self.client,
                namespace=self.config.namespace,
                embeddings_key=self.config.embeddings_key,
                embedding_dimension=vector_db.embedding_dimension,
                index_name=self.config.index_name,
                config=self.config,
            ),
            inference_api=self.inference_api,
        )
        return self.cache[vector_db_id]

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        """Unregister a vector database"""
        if vector_db_id in self.cache:
            await self.cache[vector_db_id].index.delete()
            del self.cache[vector_db_id]

    async def insert_chunks(self,
                           vector_db_id: str,
                           chunks: List[Chunk],
                           ttl_seconds: Optional[int] = None,
                           ) -> None:
        """Insert chunks into vector database"""
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")
        await index.insert_chunks(chunks)

    # async def query_chunks(self,
    #                       vector_db_id: str,
    #                       query: InterleavedContent,
    #                       params: Optional[Dict[str, Any]] = None,
    #                       ) -> QueryChunksResponse:
    #     """Query chunks from vector database"""
    #     index = await self._get_and_cache_vector_db_index(vector_db_id)
    #     if not index:
    #         raise ValueError(f"Vector DB {vector_db_id} not found")
        
    #     # Extract text from query
    #     query_text = None
    #     if self.config.search_mode != SearchMode.VECTOR:
    #         if hasattr(query, 'text'):
    #             query_text = query.text
    #         elif isinstance(query, str):
    #             query_text = query
    #         else:
    #             # Extract text from interleaved content
    #             text_parts = []
    #             if hasattr(query, '__iter__'):
    #                 for item in query:
    #                     if isinstance(item, str):
    #                         text_parts.append(item)
    #                     elif hasattr(item, 'text'):
    #                         text_parts.append(item.text)
    #             query_text = " ".join(text_parts) if text_parts else None
        
    #     # Get embeddings
    #     embeddings = await index.get_embeddings([query])
    #     embedding = embeddings[0] if embeddings else None
        
    #     # Extract parameters
    #     k = params.get('k', 10) if params else 10
    #     score_threshold = params.get('score_threshold', 0.01) if params else 0.01
    #     filters = params.get('filters') if params else None

    #     # Execute query
    #     return await index.index.query(
    #         embedding=embedding,
    #         k=k,
    #         score_threshold=score_threshold,
    #         query_text=query_text,
    #         filters=filters
    #     )

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        return await index.query_chunks(query, params)
    
    async def delete_chunks(self, store_id: str, chunks_for_deletion: List[ChunkForDeletion]) -> None:
        index = await self._get_and_cache_vector_db_index(store_id)
        if not index:
            raise ValueError(f"Vector DB {store_id} not found")
        await index.index.delete_chunks(chunks_for_deletion)

