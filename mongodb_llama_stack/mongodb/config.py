# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    VECTOR = "vector"
    FULL_TEXT = "full_text"
    HYBRID = "hybrid"
    HYBRID_GRAPH = "hybrid_graph"
    NATIVE_RANK_FUSION = "native_rank_fusion"


class GraphLookupConfig(BaseModel):
    """Configuration for MongoDB $graphLookup operations"""
    from_collection: Optional[str] = Field(None, description="Target collection (defaults to same collection)")
    start_with: str = Field("$metadata.related_ids", description="Expression for starting values")
    connect_from_field: str = Field("metadata.related_ids", description="Field to traverse from")
    connect_to_field: str = Field("metadata.document_id", description="Field to match against")
    as_field: str = Field("graph_connections", description="Output array field name")
    max_depth: int = Field(2, description="Maximum recursion depth")
    depth_field: Optional[str] = Field("graph_depth", description="Field to store depth information")
    restrict_search_with_match: Optional[Dict[str, Any]] = Field(None, description="Additional filter conditions")


class PipelineConfig(BaseModel):
    """Configuration for individual pipelines in rank fusion"""
    name: str = Field(..., description="Pipeline name")
    type: str = Field(..., description="Pipeline type: vectorSearch, search, match, geoNear")
    weight: float = Field(1.0, description="Weight for this pipeline in rank fusion")
    limit: int = Field(20, description="Maximum results from this pipeline")
    config: Dict[str, Any] = Field({}, description="Pipeline-specific configuration")


class MongoDBIOConfig(BaseModel):
    connection_str: str = Field(None, description="Connection string for MongoDB Atlas")
    namespace: str = Field(None, description="Namespace i.e. db_name.collection_name")
    
    # Index configuration
    index_name: Optional[str] = Field("default", description="Name of the vector index")
    text_index_name: Optional[str] = Field("text_index", description="Name of the text search index")
    
    # Field mappings
    embeddings_key: Optional[str] = Field("embeddings", description="Field name for embeddings")
    text_search_fields: Optional[List[str]] = Field(
        ["text", "chunk_content.content", "title"], 
        description="Fields to index for text search"
    )
    
    # Search mode configuration
    search_mode: Optional[SearchMode] = Field(SearchMode.VECTOR, description="Search mode")
    
    # Native rank fusion configuration (MongoDB 8.1+)
    rank_fusion_pipelines: Optional[List[PipelineConfig]] = Field([], description="Pipelines for rank fusion")
    enable_score_details: Optional[bool] = Field(False, description="Enable detailed scoring")
    
    # Graph lookup configuration
    graph_lookup_config: Optional[GraphLookupConfig] = Field(None, description="Graph lookup configuration")
    enable_graph_enhancement: Optional[bool] = Field(False, description="Enable graph-based result enhancement")
    
    # Fallback configuration for older MongoDB versions
    hybrid_alpha: Optional[float] = Field(0.5, description="Weight for vector search (MongoDB < 8.1)")

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **_: Any) -> Dict[str, Any]:
        return {
            "connection_str": "{env.MONGODB_CONNECTION_STR}",
            "namespace": "{env.MONGODB_NAMESPACE}",
            "search_mode": "native_rank_fusion",
            "rank_fusion_pipelines": [
                {
                    "name": "vector_pipeline",
                    "type": "vectorSearch",
                    "weight": 1.0,
                    "limit": 20,
                    "config": {
                        "numCandidates": 100,
                        "index": "vector_index"
                    }
                },
                {
                    "name": "text_pipeline",
                    "type": "search",
                    "weight": 1.0,
                    "limit": 20,
                    "config": {
                        "index": "text_index",
                        "operator": "phrase"
                    }
                }
            ],
            "graph_lookup_config": {
                "connect_from_field": "metadata.related_ids",
                "connect_to_field": "metadata.document_id",
                "max_depth": 2,
                "as_field": "related_documents"
            },
            "enable_score_details": True,
            "enable_graph_enhancement": True
        }