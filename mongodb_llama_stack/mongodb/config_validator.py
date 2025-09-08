#!/usr/bin/env python3
"""
MongoDB Provider Configuration Validator

This utility validates MongoDB provider configuration settings
and tests different search modes.
"""

import os
import argparse
import logging
from pymongo import MongoClient
import certifi
import json
from pprint import pprint
import sys
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchMode(str, Enum):
    """MongoDB search modes"""
    VECTOR = "vector"
    FULL_TEXT = "full_text"
    HYBRID = "hybrid"
    NATIVE_RANK_FUSION = "native_rank_fusion"
    HYBRID_GRAPH = "hybrid_graph"

def validate_connection():
    """Validate MongoDB connection and available features"""
    connection_str = os.environ.get("MONGODB_CONNECTION_STR")
    if not connection_str:
        logger.error("MONGODB_CONNECTION_STR environment variable not set")
        return False

    namespace = os.environ.get("MONGODB_NAMESPACE")
    if not namespace:
        logger.warning("MONGODB_NAMESPACE not set, using test.documents")
        namespace = "test.documents"
    
    db_name, collection_name = namespace.split(".")
    
    try:
        client = MongoClient(connection_str, tlsCAFile=certifi.where())
        
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        # Check server info
        server_info = client.admin.command('buildInfo')
        server_version = server_info.get('version', 'unknown')
        logger.info(f"MongoDB server version: {server_version}")
        
        # Check collection
        collection = client[db_name][collection_name]
        doc_count = collection.count_documents({})
        logger.info(f"Collection {namespace} contains {doc_count} documents")
        
        # Check indexes
        indexes = list(collection.list_indexes())
        logger.info(f"Collection has {len(indexes)} indexes")
        
        # Check search indexes
        try:
            search_indexes = list(collection.list_search_indexes())
            logger.info(f"Collection has {len(search_indexes)} search indexes")
            for idx in search_indexes:
                logger.info(f"- {idx.get('name')}")
        except Exception as e:
            logger.warning(f"Could not list search indexes: {str(e)}")
        
        return True
    
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return False

def test_search_mode(mode: SearchMode, query: str = "test"):
    """Test a specific search mode"""
    connection_str = os.environ.get("MONGODB_CONNECTION_STR")
    namespace = os.environ.get("MONGODB_NAMESPACE", "test.documents")
    db_name, collection_name = namespace.split(".")
    
    client = MongoClient(connection_str, tlsCAFile=certifi.where())
    collection = client[db_name][collection_name]
    
    logger.info(f"Testing search mode: {mode}")
    
    try:
        if mode == SearchMode.VECTOR:
            # Create simple mock embedding
            mock_embedding = [0.1] * 1536  # 1536 dimensions (adjust to your actual dimension)
            
            # Test vector search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Adjust to your index name
                        "path": "embeddings",     # Adjust to your embedding field
                        "queryVector": mock_embedding,
                        "numCandidates": 100,
                        "limit": 10
                    }
                },
                {
                    "$project": {
                        "_id": 1, 
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            
            results = list(collection.aggregate(pipeline))
            logger.info(f"Vector search returned {len(results)} results")
            if results:
                pprint(results[0])
                
        elif mode == SearchMode.FULL_TEXT:
            # Test full-text search
            pipeline = [
                {
                    "$search": {
                        "index": "text_index",  # Adjust to your index name
                        "text": {
                            "query": query,
                            "path": ["content", "title"],  # Adjust to your fields
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "title": 1,
                        "score": {"$meta": "searchScore"}
                    }
                },
                {"$limit": 10}
            ]
            
            results = list(collection.aggregate(pipeline))
            logger.info(f"Full-text search returned {len(results)} results")
            if results:
                pprint(results[0])
                
        elif mode == SearchMode.HYBRID:
            # Test hybrid search (combining vector and text search)
            mock_embedding = [0.1] * 1536
            
            pipeline = [
                {
                    "$search": {
                        "index": "default",  # Adjust to your index name
                        "compound": {
                            "should": [
                                {
                                    "text": {
                                        "query": query,
                                        "path": ["content", "title"],
                                        "score": {"boost": {"value": 1.5}}
                                    }
                                },
                                {
                                    "vectorSearch": {
                                        "queryVector": mock_embedding,
                                        "path": "embeddings",
                                        "score": {"boost": {"value": 2.0}}
                                    }
                                }
                            ]
                        }
                    }
                },
                {"$limit": 10}
            ]
            
            results = list(collection.aggregate(pipeline))
            logger.info(f"Hybrid search returned {len(results)} results")
            if results:
                pprint(results[0])
                
        elif mode == SearchMode.NATIVE_RANK_FUSION:
            # Test rank fusion (MongoDB 8.1+)
            mock_embedding = [0.1] * 1536
            
            try:
                pipeline = [
                    {
                        "$rankFusion": {
                            "clauses": [
                                {
                                    "search": {
                                        "index": "text_index",
                                        "text": {
                                            "query": query,
                                            "path": ["content", "title"]
                                        }
                                    },
                                    "weight": 1.5
                                },
                                {
                                    "search": {
                                        "index": "vector_index",
                                        "vectorSearch": {
                                            "queryVector": mock_embedding,
                                            "path": "embeddings"
                                        }
                                    },
                                    "weight": 2.0
                                }
                            ]
                        }
                    },
                    {"$limit": 10}
                ]
                
                results = list(collection.aggregate(pipeline))
                logger.info(f"Rank fusion search returned {len(results)} results")
                if results:
                    pprint(results[0])
            except Exception as e:
                logger.error(f"Rank fusion error: {str(e)}")
                logger.warning("Rank fusion requires MongoDB 8.1+")
                
        elif mode == SearchMode.HYBRID_GRAPH:
            # Test hybrid search with graph lookup
            mock_embedding = [0.1] * 1536
            
            # First get IDs with vector search
            pipeline1 = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embeddings",
                        "queryVector": mock_embedding,
                        "numCandidates": 100,
                        "limit": 5
                    }
                },
                {"$project": {"_id": 1}}
            ]
            
            seed_results = list(collection.aggregate(pipeline1))
            seed_ids = [r["_id"] for r in seed_results]
            
            if seed_ids:
                # Now find related documents using graph traversal
                pipeline2 = [
                    {"$match": {"_id": {"$in": seed_ids}}},
                    {
                        "$graphLookup": {
                            "from": collection_name,
                            "startWith": "$metadata.related_ids",
                            "connectFromField": "metadata.related_ids",
                            "connectToField": "_id",
                            "as": "related_docs",
                            "maxDepth": 1
                        }
                    },
                    {"$limit": 10}
                ]
                
                results = list(collection.aggregate(pipeline2))
                logger.info(f"Graph-enhanced search returned {len(results)} results")
                if results:
                    pprint(results[0])
            else:
                logger.warning("No seed documents found for graph search")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing {mode}: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate MongoDB provider configuration")
    parser.add_argument("--mode", choices=[m.value for m in SearchMode], help="Search mode to test")
    parser.add_argument("--query", default="test", help="Query text for search")
    args = parser.parse_args()
    
    # Validate connection
    if not validate_connection():
        logger.error("Connection validation failed")
        sys.exit(1)
    
    # Test search mode if specified
    if args.mode:
        mode = SearchMode(args.mode)
        if test_search_mode(mode, args.query):
            logger.info(f"Successfully tested {mode}")
        else:
            logger.error(f"Failed testing {mode}")
            sys.exit(1)

if __name__ == "__main__":
    main()
