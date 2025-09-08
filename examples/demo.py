#!/usr/bin/env python3
"""
Demo script showcasing MongoDB Atlas Vector Search with Llama Stack
This script demonstrates vector search, text search, and hybrid search capabilities
"""

import asyncio
import os
import sys
import json
import logging
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from mongodb_llama_stack.mongodb.mongodb import MongoDBIOAdapter
    from mongodb_llama_stack.mongodb.config import MongoDBIOConfig, SearchMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the MongoDB provider is installed: pip install -e .")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class DemoInferenceAPI:
    """Demo inference API with sample embeddings"""
    
    async def embeddings(self, content_batch, model=None):
        """Generate demo embeddings using simple text features"""
        import hashlib
        
        class DemoEmbeddingResponse:
            def __init__(self, texts):
                self.embeddings = []
                for text in texts:
                    # Generate deterministic embeddings based on text content
                    embedding = self._text_to_embedding(text)
                    self.embeddings.append(embedding)
        
            def _text_to_embedding(self, text: str, dim: int = 384):
                """Convert text to a simple embedding vector"""
                # Use text hash to generate consistent embeddings
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                # Convert hash to numbers and normalize
                embedding = []
                for i in range(0, min(len(text_hash), dim//16)):
                    chunk = text_hash[i*2:(i+1)*2]
                    val = int(chunk, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
                    embedding.extend([val] * 16)
                
                # Pad or truncate to exact dimension
                while len(embedding) < dim:
                    embedding.append(0.0)
                embedding = embedding[:dim]
                
                return embedding
        
        return DemoEmbeddingResponse([item for item in content_batch])

class MongoDBDemo:
    """MongoDB Atlas Vector Search Demo"""
    
    def __init__(self):
        self.connection_str = os.getenv("MONGODB_CONNECTION_STR")
        self.namespace = os.getenv("MONGODB_NAMESPACE", "demo_llama.vector_search")
        self.inference_api = DemoInferenceAPI()
        
        if not self.connection_str:
            raise ValueError("MONGODB_CONNECTION_STR environment variable is required")
    
    async def demo_vector_search(self):
        """Demonstrate vector search functionality"""
        print("\n🔍 VECTOR SEARCH DEMO")
        print("=" * 50)
        
        config = MongoDBIOConfig(
            connection_str=self.connection_str,
            namespace=f"{self.namespace}_vector",
            search_mode=SearchMode.VECTOR,
            embeddings_key="embeddings",
            index_name="demo_vector_index"
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        await adapter.initialize()
        
        # Register vector database
        from mongodb_llama_stack.mongodb.mongodb import VectorDB, Chunk
        vector_db = VectorDB(
            identifier="demo_vector_db",
            embedding_dimension=384,
            embedding_type="dense"
        )
        await adapter.register_vector_db(vector_db)
        
        # Sample documents about databases and AI
        documents = [
            {
                "content": "MongoDB is a NoSQL document database that provides high performance, high availability, and easy scalability.",
                "metadata": {"title": "MongoDB Overview", "category": "database", "doc_id": "mongo_1"}
            },
            {
                "content": "Vector databases are specialized systems designed to store and query high-dimensional vectors efficiently.",
                "metadata": {"title": "Vector Databases", "category": "database", "doc_id": "vector_1"}
            },
            {
                "content": "Machine learning models generate embeddings that capture semantic meaning of text, images, or other data.",
                "metadata": {"title": "ML Embeddings", "category": "ai", "doc_id": "ml_1"}
            },
            {
                "content": "Atlas Search provides full-text search capabilities with powerful query syntax and relevance scoring.",
                "metadata": {"title": "Atlas Search", "category": "search", "doc_id": "atlas_1"}
            },
            {
                "content": "Semantic search uses vector similarity to find documents based on meaning rather than exact keyword matches.",
                "metadata": {"title": "Semantic Search", "category": "search", "doc_id": "semantic_1"}
            }
        ]
        
        # Insert documents
        chunks = [Chunk(content=doc["content"], metadata=doc["metadata"]) for doc in documents]
        print(f"📝 Inserting {len(chunks)} documents...")
        await adapter.insert_chunks("demo_vector_db", chunks)
        
        # Perform vector searches
        queries = [
            "database systems for storing data",
            "machine learning and artificial intelligence",
            "search technologies and algorithms"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = await adapter.query_chunks(
                "demo_vector_db",
                query,
                params={"max_chunks": 3, "score_threshold": 0.1}
            )
            
            if results and results.chunks:
                for i, (chunk, score) in enumerate(zip(results.chunks, results.scores), 1):
                    print(f"   {i}. [{score:.3f}] {chunk.metadata.get('title', 'Unknown')}")
                    print(f"      {chunk.content[:100]}...")
            else:
                print("   No results found")
        
        await adapter.shutdown()
        print("✅ Vector search demo completed")
    
    async def demo_text_search(self):
        """Demonstrate text search functionality"""
        print("\n📄 TEXT SEARCH DEMO")
        print("=" * 50)
        
        config = MongoDBIOConfig(
            connection_str=self.connection_str,
            namespace=f"{self.namespace}_text",
            search_mode=SearchMode.FULL_TEXT,
            text_index_name="demo_text_index",
            text_search_fields=["content", "metadata.title", "metadata.category"]
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        await adapter.initialize()
        
        # Register vector database
        from mongodb_llama_stack.mongodb.mongodb import VectorDB, Chunk
        vector_db = VectorDB(
            identifier="demo_text_db",
            embedding_dimension=384,
            embedding_type="dense"
        )
        await adapter.register_vector_db(vector_db)
        
        # Sample documents with rich text content
        documents = [
            {
                "content": "MongoDB Atlas provides powerful full-text search capabilities using Lucene-based indexing.",
                "metadata": {"title": "Atlas Text Search", "category": "database", "tags": ["mongodb", "search", "atlas"]}
            },
            {
                "content": "Full-text search enables users to search for documents containing specific words or phrases.",
                "metadata": {"title": "Full-Text Search Basics", "category": "search", "tags": ["fulltext", "query", "documents"]}
            },
            {
                "content": "Elasticsearch and Solr are popular search engines for full-text search and analytics.",
                "metadata": {"title": "Search Engine Technologies", "category": "technology", "tags": ["elasticsearch", "solr", "analytics"]}
            },
            {
                "content": "Text analysis and natural language processing help improve search relevance and user experience.",
                "metadata": {"title": "NLP for Search", "category": "ai", "tags": ["nlp", "analysis", "relevance"]}
            }
        ]
        
        # Insert documents
        chunks = [Chunk(content=doc["content"], metadata=doc["metadata"]) for doc in documents]
        print(f"📝 Inserting {len(chunks)} documents...")
        await adapter.insert_chunks("demo_text_db", chunks)
        
        # Perform text searches
        queries = [
            "MongoDB Atlas search",
            "full-text search documents",
            "natural language processing",
            "Elasticsearch analytics"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = await adapter.query_chunks(
                "demo_text_db",
                query,
                params={"max_chunks": 3}
            )
            
            if results and results.chunks:
                for i, chunk in enumerate(results.chunks, 1):
                    print(f"   {i}. {chunk.metadata.get('title', 'Unknown')}")
                    print(f"      {chunk.content[:100]}...")
            else:
                print("   No results found")
        
        await adapter.shutdown()
        print("✅ Text search demo completed")
    
    async def demo_hybrid_search(self):
        """Demonstrate hybrid search functionality"""
        print("\n🔄 HYBRID SEARCH DEMO")
        print("=" * 50)
        
        config = MongoDBIOConfig(
            connection_str=self.connection_str,
            namespace=f"{self.namespace}_hybrid",
            search_mode=SearchMode.HYBRID,
            embeddings_key="embeddings",
            index_name="demo_vector_index",
            text_index_name="demo_text_index",
            text_search_fields=["content", "metadata.title"],
            hybrid_alpha=0.6  # 60% vector, 40% text
        )
        
        adapter = MongoDBIOAdapter(config, self.inference_api)
        await adapter.initialize()
        
        # Register vector database
        from mongodb_llama_stack.mongodb.mongodb import VectorDB, Chunk
        vector_db = VectorDB(
            identifier="demo_hybrid_db",
            embedding_dimension=384,
            embedding_type="dense"
        )
        await adapter.register_vector_db(vector_db)
        
        # Sample documents combining technical and business content
        documents = [
            {
                "content": "Artificial intelligence and machine learning are transforming how businesses analyze customer data and behavior patterns.",
                "metadata": {"title": "AI in Business Analytics", "category": "business", "domain": "analytics"}
            },
            {
                "content": "Vector databases enable semantic search by storing high-dimensional embeddings and computing similarity scores.",
                "metadata": {"title": "Vector Database Technology", "category": "technology", "domain": "database"}
            },
            {
                "content": "Customer experience optimization uses AI-powered recommendation systems to personalize user interactions.",
                "metadata": {"title": "AI-Powered Recommendations", "category": "business", "domain": "customer_experience"}
            },
            {
                "content": "Natural language processing techniques help extract insights from unstructured text data and social media.",
                "metadata": {"title": "NLP for Data Analysis", "category": "technology", "domain": "analytics"}
            },
            {
                "content": "MongoDB Atlas combines document storage with powerful search and analytics capabilities for modern applications.",
                "metadata": {"title": "MongoDB Atlas Platform", "category": "database", "domain": "platform"}
            }
        ]
        
        # Insert documents
        chunks = [Chunk(content=doc["content"], metadata=doc["metadata"]) for doc in documents]
        print(f"📝 Inserting {len(chunks)} documents...")
        await adapter.insert_chunks("demo_hybrid_db", chunks)
        
        # Perform hybrid searches
        queries = [
            "AI machine learning customer analytics",
            "vector database similarity search",
            "MongoDB document storage platform",
            "natural language processing insights"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            results = await adapter.query_chunks(
                "demo_hybrid_db",
                query,
                params={"max_chunks": 3, "mode": "hybrid"}
            )
            
            if results and results.chunks:
                for i, (chunk, score) in enumerate(zip(results.chunks, results.scores), 1):
                    print(f"   {i}. [{score:.3f}] {chunk.metadata.get('title', 'Unknown')}")
                    print(f"      Domain: {chunk.metadata.get('domain', 'N/A')}")
                    print(f"      {chunk.content[:80]}...")
            else:
                print("   No results found")
        
        await adapter.shutdown()
        print("✅ Hybrid search demo completed")
    
    async def run_all_demos(self):
        """Run all demo scenarios"""
        print("🚀 MongoDB Atlas Vector Search Demo")
        print("====================================")
        print(f"Connection: {self.connection_str[:30]}...")
        print(f"Namespace: {self.namespace}")
        
        try:
            await self.demo_vector_search()
            await self.demo_text_search()
            await self.demo_hybrid_search()
            
            print("\n🎉 All demos completed successfully!")
            print("\nNext steps:")
            print("1. Try with your own documents and queries")
            print("2. Experiment with different search modes")
            print("3. Adjust hybrid_alpha for different vector/text weights")
            print("4. Configure custom index settings for your use case")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            raise

async def main():
    """Main demo runner"""
    # Check environment
    if not os.getenv("MONGODB_CONNECTION_STR"):
        print("❌ Error: MONGODB_CONNECTION_STR environment variable is required")
        print("\nExample setup:")
        print("export MONGODB_CONNECTION_STR='mongodb+srv://user:pass@cluster.mongodb.net/'")
        print("export MONGODB_NAMESPACE='demo_llama.vector_search'")
        print("\nThen run: python examples/demo.py")
        sys.exit(1)
    
    # Run demos
    demo = MongoDBDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())
