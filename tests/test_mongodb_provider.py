#!/usr/bin/env python3
"""
Comprehensive test suite for MongoDB Atlas Vector Search, Hybrid Search, and Text Search integrations
"""

import asyncio
import os
import pytest
import logging
from typing import List
from unittest.mock import Mock, AsyncMock

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk
from llama_stack.apis.inference import InferenceAPI
from llama_stack.providers.remote.vector_io.mongodb import MongoDBIOAdapter
from llama_stack.providers.remote.vector_io.mongodb.config import MongoDBIOConfig, SearchMode

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Test constants
TEST_NAMESPACE = "test_llama.test_collection"
TEST_DB_NAME = "test_mongodb_provider"
EMBEDDING_DIMENSION = 384

class TestMongoDBProvider:
    """Test suite for MongoDB Atlas Vector Search provider"""
    
    @pytest.fixture
    def mock_inference_api(self):
        """Mock inference API for testing"""
        mock_api = Mock(spec=InferenceAPI)
        mock_api.embeddings = AsyncMock()
        mock_api.embeddings.return_value = Mock(
            embeddings=[[0.1] * EMBEDDING_DIMENSION for _ in range(3)]
        )
        return mock_api
    
    @pytest.fixture
    def sample_chunks(self) -> List[Chunk]:
        """Sample chunks for testing"""
        return [
            Chunk(
                content="MongoDB is a document database with vector search capabilities",
                metadata={
                    "doc_id": "doc1",
                    "title": "MongoDB Introduction",
                    "category": "database",
                    "tags": ["mongodb", "vector", "search"]
                }
            ),
            Chunk(
                content="Artificial intelligence and machine learning are transforming technology",
                metadata={
                    "doc_id": "doc2", 
                    "title": "AI and ML Overview",
                    "category": "technology",
                    "tags": ["ai", "ml", "technology"]
                }
            ),
            Chunk(
                content="Vector databases enable semantic search and similarity matching",
                metadata={
                    "doc_id": "doc3",
                    "title": "Vector Database Concepts", 
                    "category": "database",
                    "tags": ["vector", "semantic", "similarity"]
                }
            )
        ]
    
    @pytest.fixture
    def vector_db_config(self):
        """Vector database configuration for testing"""
        return VectorDB(
            identifier=TEST_DB_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
            embedding_type="dense"
        )

class TestVectorSearch(TestMongoDBProvider):
    """Test vector search functionality"""
    
    @pytest.fixture
    def vector_config(self):
        """Configuration for vector search testing"""
        return MongoDBIOConfig(
            connection_str=os.getenv("MONGODB_CONNECTION_STR", "mongodb://localhost:27017"),
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR,
            embeddings_key="embeddings",
            index_name="test_vector_index"
        )
    
    @pytest.mark.asyncio
    async def test_vector_search_initialization(self, vector_config, mock_inference_api):
        """Test vector search adapter initialization"""
        adapter = MongoDBIOAdapter(vector_config, mock_inference_api)
        await adapter.initialize()
        
        assert adapter.config.search_mode == SearchMode.VECTOR
        assert adapter.config.embeddings_key == "embeddings"
        
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_vector_db_registration(self, vector_config, mock_inference_api, vector_db_config):
        """Test vector database registration"""
        adapter = MongoDBIOAdapter(vector_config, mock_inference_api)
        await adapter.initialize()
        
        await adapter.register_vector_db(vector_db_config)
        
        # Verify the database is registered
        assert TEST_DB_NAME in adapter.vector_dbs
        
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_vector_chunk_insertion(self, vector_config, mock_inference_api, vector_db_config, sample_chunks):
        """Test inserting chunks with vector embeddings"""
        adapter = MongoDBIOAdapter(vector_config, mock_inference_api)
        await adapter.initialize()
        await adapter.register_vector_db(vector_db_config)
        
        # Insert chunks
        await adapter.insert_chunks(TEST_DB_NAME, sample_chunks)
        
        # Mock should have been called to generate embeddings
        mock_inference_api.embeddings.assert_called()
        
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_vector_search_query(self, vector_config, mock_inference_api, vector_db_config, sample_chunks):
        """Test vector similarity search"""
        adapter = MongoDBIOAdapter(vector_config, mock_inference_api)
        await adapter.initialize()
        await adapter.register_vector_db(vector_db_config)
        
        # Insert test data
        await adapter.insert_chunks(TEST_DB_NAME, sample_chunks)
        
        # Query for similar documents
        query = "database vector search"
        results = await adapter.query_chunks(
            TEST_DB_NAME,
            query,
            params={"max_chunks": 5, "score_threshold": 0.1}
        )
        
        assert results is not None
        assert hasattr(results, 'chunks')
        
        await adapter.shutdown()

class TestTextSearch(TestMongoDBProvider):
    """Test text search functionality"""
    
    @pytest.fixture
    def text_config(self):
        """Configuration for text search testing"""
        return MongoDBIOConfig(
            connection_str=os.getenv("MONGODB_CONNECTION_STR", "mongodb://localhost:27017"),
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.FULL_TEXT,
            text_index_name="test_text_index",
            text_search_fields=["title", "content", "category"]
        )
    
    @pytest.mark.asyncio
    async def test_text_search_initialization(self, text_config, mock_inference_api):
        """Test text search adapter initialization"""
        adapter = MongoDBIOAdapter(text_config, mock_inference_api)
        await adapter.initialize()
        
        assert adapter.config.search_mode == SearchMode.FULL_TEXT
        assert "title" in adapter.config.text_search_fields
        
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_text_search_query(self, text_config, mock_inference_api, vector_db_config, sample_chunks):
        """Test full-text search functionality"""
        adapter = MongoDBIOAdapter(text_config, mock_inference_api)
        await adapter.initialize()
        await adapter.register_vector_db(vector_db_config)
        
        # Insert test data
        await adapter.insert_chunks(TEST_DB_NAME, sample_chunks)
        
        # Query for text matches
        query = "mongodb database"
        results = await adapter.query_chunks(
            TEST_DB_NAME,
            query,
            params={"max_chunks": 5}
        )
        
        assert results is not None
        
        await adapter.shutdown()

class TestHybridSearch(TestMongoDBProvider):
    """Test hybrid search functionality"""
    
    @pytest.fixture
    def hybrid_config(self):
        """Configuration for hybrid search testing"""
        return MongoDBIOConfig(
            connection_str=os.getenv("MONGODB_CONNECTION_STR", "mongodb://localhost:27017"),
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.HYBRID,
            embeddings_key="embeddings",
            index_name="test_vector_index",
            text_index_name="test_text_index",
            text_search_fields=["title", "content", "category"],
            hybrid_alpha=0.7
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_initialization(self, hybrid_config, mock_inference_api):
        """Test hybrid search adapter initialization"""
        adapter = MongoDBIOAdapter(hybrid_config, mock_inference_api)
        await adapter.initialize()
        
        assert adapter.config.search_mode == SearchMode.HYBRID
        assert adapter.config.hybrid_alpha == 0.7
        
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_query(self, hybrid_config, mock_inference_api, vector_db_config, sample_chunks):
        """Test hybrid search combining vector and text"""
        adapter = MongoDBIOAdapter(hybrid_config, mock_inference_api)
        await adapter.initialize()
        await adapter.register_vector_db(vector_db_config)
        
        # Insert test data
        await adapter.insert_chunks(TEST_DB_NAME, sample_chunks)
        
        # Query combining semantic and keyword search
        query = "artificial intelligence database technology"
        results = await adapter.query_chunks(
            TEST_DB_NAME,
            query,
            params={"max_chunks": 5, "mode": "hybrid"}
        )
        
        assert results is not None
        
        await adapter.shutdown()

class TestNativeRankFusion(TestMongoDBProvider):
    """Test native rank fusion functionality (MongoDB 8.1+)"""
    
    @pytest.fixture
    def rank_fusion_config(self):
        """Configuration for native rank fusion testing"""
        return MongoDBIOConfig(
            connection_str=os.getenv("MONGODB_CONNECTION_STR", "mongodb://localhost:27017"),
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.NATIVE_RANK_FUSION,
            rank_fusion_pipelines=[
                {
                    "name": "vector_pipeline",
                    "type": "vectorSearch",
                    "weight": 1.5,
                    "limit": 20,
                    "config": {
                        "numCandidates": 100,
                        "index": "test_vector_index"
                    }
                },
                {
                    "name": "text_pipeline", 
                    "type": "search",
                    "weight": 1.0,
                    "limit": 20,
                    "config": {
                        "index": "test_text_index",
                        "operator": "phrase"
                    }
                }
            ],
            enable_score_details=True
        )
    
    @pytest.mark.asyncio
    async def test_rank_fusion_initialization(self, rank_fusion_config, mock_inference_api):
        """Test rank fusion adapter initialization"""
        adapter = MongoDBIOAdapter(rank_fusion_config, mock_inference_api)
        await adapter.initialize()
        
        assert adapter.config.search_mode == SearchMode.NATIVE_RANK_FUSION
        assert len(adapter.config.rank_fusion_pipelines) == 2
        
        await adapter.shutdown()

class TestErrorHandling(TestMongoDBProvider):
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_connection_string(self, mock_inference_api):
        """Test handling of invalid connection strings"""
        config = MongoDBIOConfig(
            connection_str="invalid://connection:string",
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR
        )
        
        adapter = MongoDBIOAdapter(config, mock_inference_api)
        
        with pytest.raises(Exception):
            await adapter.initialize()
    
    @pytest.mark.asyncio
    async def test_missing_vector_db(self, mock_inference_api):
        """Test querying non-existent vector database"""
        config = MongoDBIOConfig(
            connection_str=os.getenv("MONGODB_CONNECTION_STR", "mongodb://localhost:27017"),
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR
        )
        
        adapter = MongoDBIOAdapter(config, mock_inference_api)
        await adapter.initialize()
        
        with pytest.raises(ValueError):
            await adapter.query_chunks("nonexistent_db", "test query")
        
        await adapter.shutdown()

class TestPerformance(TestMongoDBProvider):
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_batch_insertion_performance(self, mock_inference_api, vector_db_config):
        """Test performance with large batch insertions"""
        config = MongoDBIOConfig(
            connection_str=os.getenv("MONGODB_CONNECTION_STR", "mongodb://localhost:27017"),
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR
        )
        
        adapter = MongoDBIOAdapter(config, mock_inference_api)
        await adapter.initialize()
        await adapter.register_vector_db(vector_db_config)
        
        # Create large batch of chunks
        large_batch = []
        for i in range(100):
            chunk = Chunk(
                content=f"Test document content number {i} with various keywords",
                metadata={"doc_id": f"doc_{i}", "batch": "performance_test"}
            )
            large_batch.append(chunk)
        
        # Insert large batch
        await adapter.insert_chunks(TEST_DB_NAME, large_batch)
        
        # Verify insertion
        results = await adapter.query_chunks(
            TEST_DB_NAME,
            "test document",
            params={"max_chunks": 10}
        )
        
        assert results is not None
        assert len(results.chunks) > 0
        
        await adapter.shutdown()

if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__, "-v"])
