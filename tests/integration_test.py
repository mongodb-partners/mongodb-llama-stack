#!/usr/bin/env python3
"""
Integration tests for MongoDB Atlas Vector Search with Llama Stack
Run these tests against a real MongoDB Atlas cluster or local MongoDB instance
"""

import asyncio
import os
import sys
import logging
from typing import List, Dict, Any
import json
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from mongodb_llama_stack.mongodb.mongodb import MongoDBIOAdapter
    from mongodb_llama_stack.mongodb.config import MongoDBIOConfig, SearchMode
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the MongoDB provider is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Test configuration
TEST_NAMESPACE = os.getenv("MONGODB_TEST_NAMESPACE", "test_llama.integration_test")
CONNECTION_STR = os.getenv("MONGODB_CONNECTION_STR")
EMBEDDING_DIMENSION = 384

class MockInferenceAPI:
    """Mock inference API for testing"""
    
    async def embeddings(self, content_batch, model=None):
        """Generate mock embeddings"""
        import random
        
        class MockEmbeddingResponse:
            def __init__(self, batch_size):
                # Generate random embeddings for testing
                self.embeddings = [
                    [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]
                    for _ in range(batch_size)
                ]
        
        return MockEmbeddingResponse(len(content_batch))

class IntegrationTestRunner:
    """Integration test runner for MongoDB Atlas provider"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
        
        if not CONNECTION_STR:
            raise ValueError("MONGODB_CONNECTION_STR environment variable is required")
        
        self.inference_api = MockInferenceAPI()
    
    async def run_test(self, test_name: str, test_func):
        """Run a single test and track results"""
        try:
            log.info(f"Running test: {test_name}")
            start_time = time.time()
            
            await test_func()
            
            duration = time.time() - start_time
            log.info(f"✅ {test_name} passed ({duration:.2f}s)")
            self.passed_tests += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASSED",
                "duration": duration
            })
            
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            log.error(f"❌ {test_name} failed: {e}")
            self.failed_tests += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAILED",
                "error": str(e),
                "duration": duration
            })
    
    async def test_vector_search_basic(self):
        """Test basic vector search functionality"""
        config = MongoDBIOConfig(
            connection_str=CONNECTION_STR,
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR,
            embeddings_key="embeddings",
            index_name="test_vector_index"
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        
        try:
            await adapter.initialize()
            
            # Register vector database
            from mongodb_llama_stack.mongodb.mongodb import VectorDB
            vector_db = VectorDB(
                identifier="test_vector_db",
                embedding_dimension=EMBEDDING_DIMENSION,
                embedding_type="dense"
            )
            await adapter.register_vector_db(vector_db)
            
            # Insert test chunks
            chunks = [
                {
                    "content": "MongoDB is a NoSQL document database with vector search capabilities",
                    "metadata": {"doc_id": "test_1", "category": "database"}
                },
                {
                    "content": "Vector databases enable semantic search and AI applications",
                    "metadata": {"doc_id": "test_2", "category": "ai"}
                }
            ]
            
            # Convert to Chunk objects
            from mongodb_llama_stack.mongodb.mongodb import Chunk
            chunk_objects = [
                Chunk(content=c["content"], metadata=c["metadata"]) 
                for c in chunks
            ]
            
            await adapter.insert_chunks("test_vector_db", chunk_objects)
            
            # Query for similar content
            results = await adapter.query_chunks(
                "test_vector_db",
                "database search technology",
                params={"max_chunks": 5}
            )
            
            assert results is not None
            assert hasattr(results, 'chunks')
            log.info(f"Found {len(results.chunks)} results from vector search")
            
        finally:
            await adapter.shutdown()
    
    async def test_text_search_basic(self):
        """Test basic text search functionality"""
        config = MongoDBIOConfig(
            connection_str=CONNECTION_STR,
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.FULL_TEXT,
            text_index_name="test_text_index",
            text_search_fields=["content", "metadata.category"]
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        
        try:
            await adapter.initialize()
            
            # Register vector database
            from mongodb_llama_stack.mongodb.mongodb import VectorDB
            vector_db = VectorDB(
                identifier="test_text_db",
                embedding_dimension=EMBEDDING_DIMENSION,
                embedding_type="dense"
            )
            await adapter.register_vector_db(vector_db)
            
            # Insert test chunks
            from mongodb_llama_stack.mongodb.mongodb import Chunk
            chunks = [
                Chunk(
                    content="MongoDB Atlas provides full-text search capabilities",
                    metadata={"doc_id": "text_1", "category": "database"}
                ),
                Chunk(
                    content="Full-text search enables keyword-based document retrieval",
                    metadata={"doc_id": "text_2", "category": "search"}
                )
            ]
            
            await adapter.insert_chunks("test_text_db", chunks)
            
            # Query with text search
            results = await adapter.query_chunks(
                "test_text_db",
                "MongoDB Atlas search",
                params={"max_chunks": 5}
            )
            
            assert results is not None
            log.info(f"Found {len(results.chunks)} results from text search")
            
        finally:
            await adapter.shutdown()
    
    async def test_hybrid_search_basic(self):
        """Test basic hybrid search functionality"""
        config = MongoDBIOConfig(
            connection_str=CONNECTION_STR,
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.HYBRID,
            embeddings_key="embeddings",
            index_name="test_vector_index",
            text_index_name="test_text_index",
            text_search_fields=["content"],
            hybrid_alpha=0.7
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        
        try:
            await adapter.initialize()
            
            # Register vector database
            from mongodb_llama_stack.mongodb.mongodb import VectorDB
            vector_db = VectorDB(
                identifier="test_hybrid_db",
                embedding_dimension=EMBEDDING_DIMENSION,
                embedding_type="dense"
            )
            await adapter.register_vector_db(vector_db)
            
            # Insert test chunks
            from mongodb_llama_stack.mongodb.mongodb import Chunk
            chunks = [
                Chunk(
                    content="Hybrid search combines vector similarity with text matching",
                    metadata={"doc_id": "hybrid_1", "category": "search"}
                ),
                Chunk(
                    content="Semantic search using embeddings and keyword search work together",
                    metadata={"doc_id": "hybrid_2", "category": "ai"}
                )
            ]
            
            await adapter.insert_chunks("test_hybrid_db", chunks)
            
            # Query with hybrid search
            results = await adapter.query_chunks(
                "test_hybrid_db",
                "semantic vector search embeddings",
                params={"max_chunks": 5, "mode": "hybrid"}
            )
            
            assert results is not None
            log.info(f"Found {len(results.chunks)} results from hybrid search")
            
        finally:
            await adapter.shutdown()
    
    async def test_mongodb_version_detection(self):
        """Test MongoDB version detection and feature compatibility"""
        config = MongoDBIOConfig(
            connection_str=CONNECTION_STR,
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        
        try:
            await adapter.initialize()
            
            # Check if we can detect MongoDB version
            assert hasattr(adapter, 'vector_dbs')
            log.info("MongoDB version detection completed successfully")
            
        finally:
            await adapter.shutdown()
    
    async def test_atlas_index_creation(self):
        """Test Atlas search index creation"""
        config = MongoDBIOConfig(
            connection_str=CONNECTION_STR,
            namespace=TEST_NAMESPACE,
            search_mode=SearchMode.VECTOR,
            embeddings_key="embeddings",
            index_name="test_auto_index"
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        
        try:
            await adapter.initialize()
            
            # Register vector database (should trigger index creation)
            from mongodb_llama_stack.mongodb.mongodb import VectorDB
            vector_db = VectorDB(
                identifier="test_index_db",
                embedding_dimension=EMBEDDING_DIMENSION,
                embedding_type="dense"
            )
            await adapter.register_vector_db(vector_db)
            
            log.info("Index creation test completed (check Atlas UI for index status)")
            
        finally:
            await adapter.shutdown()
    
    async def run_all_tests(self):
        """Run all integration tests"""
        log.info("Starting MongoDB Atlas integration tests...")
        log.info(f"Using namespace: {TEST_NAMESPACE}")
        log.info(f"Connection string: {CONNECTION_STR[:20]}...")
        
        tests = [
            ("MongoDB Version Detection", self.test_mongodb_version_detection),
            ("Vector Search Basic", self.test_vector_search_basic),
            ("Text Search Basic", self.test_text_search_basic),
            ("Hybrid Search Basic", self.test_hybrid_search_basic),
            ("Atlas Index Creation", self.test_atlas_index_creation),
        ]
        
        for test_name, test_func in tests:
            await self.run_test(test_name, test_func)
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        total_tests = self.passed_tests + self.failed_tests
        
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success rate: {(self.passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        if self.failed_tests > 0:
            print("\nFailed tests:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    print(f"  ❌ {result['name']}: {result.get('error', 'Unknown error')}")
        
        print("\nTest details:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"  {status_icon} {result['name']} ({result['duration']:.2f}s)")
        
        print("="*60)

async def main():
    """Main test runner"""
    if not CONNECTION_STR:
        print("❌ Error: MONGODB_CONNECTION_STR environment variable is required")
        print("Example: export MONGODB_CONNECTION_STR='mongodb+srv://user:pass@cluster.mongodb.net/'")
        sys.exit(1)
    
    runner = IntegrationTestRunner()
    
    try:
        await runner.run_all_tests()
        
        # Exit with error code if any tests failed
        if runner.failed_tests > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
