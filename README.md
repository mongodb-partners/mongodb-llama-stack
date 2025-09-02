# MongoDB Vector IO Provider for Llama Stack

A MongoDB integration for Llama Stack that supports vector search, full‑text search, hybrid search, and graph‑enhanced retrieval with native MongoDB features.

## What you get

- Native MongoDB 8.1+ features: `$rankFusion`, `$graphLookup`
- Multiple search modes: vector, full-text, hybrid, hybrid+graph
- Auto-detection of server features with safe fallbacks for older versions
- Practical defaults, robust logging, and index auto-provisioning (Atlas)

## Requirements

- Python 3.10+
- MongoDB Atlas cluster or MongoDB 8.0+
- Llama Stack installed and runnable

---

## Quick start (pick one)

You can load this provider into Llama Stack in two ways. Use Path A for local development (no copying), or Path B when installing as a module.

### Path A — External providers directory (local dev)
1) Install dependencies and this package locally
```bash
pip install pymongo certifi numpy packaging
pip install -e .
```

2) Export environment variables
```bash
export MONGODB_CONNECTION_STR='mongodb+srv://<user>:<pass>@<host>/'
export MONGODB_NAMESPACE='mydb.mycollection'
export EXTERNAL_PROVIDERS_DIR="$(pwd)/providers.d"
```

3) Reference the provider in your run.yaml and include the external providers dir
```yaml
version: '2'
apis:
    - vector_io
providers:
    vector_io:
        - provider_id: mongodb
            provider_type: remote::mongodb
            config:
                connection_str: ${env.MONGODB_CONNECTION_STR:+}
                namespace: ${env.MONGODB_NAMESPACE:+}
                # Optional:
                # search_mode: vector | full_text | hybrid | native_rank_fusion | hybrid_graph
                # index_name: default
                # text_index_name: text_index
                # text_search_fields: ["title", "content", "description"]
external_providers_dir: ${env.EXTERNAL_PROVIDERS_DIR:=~/.llama/providers.d}
```

4) Build and run
```bash
llama stack build
llama stack run
```

Notes
- Example config: `mongodb_llama_stack/run.yaml` shows how to add `remote::mongodb` to `vector_io`.
- Discovery spec: `providers.d/remote/vector_io/mongodb.yaml`
- Implementation: `providers.d/remote/vector_io/mongodb/`

### Path B — Module (packaged install)
1) Install the package (from PyPI or your build)
```bash
pip install mongodb-llama-stack
```

2) Reference the module in your config (build.yaml or run.yaml)
```yaml
providers:
    vector_io:
        - provider_type: remote::mongodb
            module: mongodb_llama_stack
            # config keys are identical to Path A
```

3) Build and run
```bash
llama stack build
llama stack run
```

The provider is discovered as `remote::mongodb` in both paths.

---


## MongoDB setup

### MongoDB Atlas (Recommended)
1. Create a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register)
2. Create a new cluster (M0 free tier works for testing)
3. Get your connection string from Atlas Dashboard
4. Ensure your IP is whitelisted in Network Access

### Self-hosted MongoDB
Ensure you're running MongoDB 8.0+ for full feature support:
```bash
mongosh --eval "db.version()"
```

---


## Configuration

### Basic Configuration
```python
from llama_stack.providers.remote.vector_io.mongodb import MongoDBIOConfig
config = MongoDBIOConfig(
    connection_str="mongodb+srv://username:password@cluster.mongodb.net/",
    namespace="mydb.mycollection"
)
```

### Environment Variables
Create a `.env` file:
```bash
MONGODB_CONNECTION_STR=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_NAMESPACE=mydb.mycollection
```

---


## Search Modes

### 1. Vector Search Only

Best for semantic similarity search using embeddings.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode="vector",
    embeddings_key="plot_embedding",  # Field containing embeddings
    index_name="vector_index"
)
```

### 2. Full-Text Search Only

Best for keyword-based search with exact matches.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode="full_text",
    text_index_name="text_index",
    text_search_fields=["title", "content", "description"]
)
```

### 3. Hybrid Search (Vector + Text)

Combines semantic and keyword search for best results.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode="hybrid",
    embeddings_key="embeddings",
    text_search_fields=["title", "content"],
    hybrid_alpha=0.7  # 70% weight to vector, 30% to text
)
```

### 4. Native Rank Fusion (MongoDB 8.1+)

Leverages MongoDB's native `$rankFusion` for optimal performance.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode="native_rank_fusion",
    rank_fusion_pipelines=[
        {
            "name": "vector_pipeline",
            "type": "vectorSearch",
            "weight": 1.5,  # Higher weight for vector search
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
                "operator": "phrase"  # or "text", "compound"
            }
        }
    ],
    enable_score_details=True  # Get detailed scoring information
)
```

### 5. Graph-Enhanced Search (TBD)
This feature will be included as part of the repo shortly
<!-- ### 5. Graph-Enhanced Search

Includes related documents through graph traversal using MongoDB's `$graphLookup` operator.

```python
from llama_stack.providers.remote.vector_io.mongodb import (
    MongoDBIOConfig,
    GraphLookupConfig,
    SearchMode
)

config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode="hybrid_graph",
    
    # Properly typed GraphLookupConfig
    graph_lookup_config=GraphLookupConfig(
        from_collection=None,  # None = same collection, or specify another
        start_with="$metadata.related_ids",  # Expression for starting values
        connect_from_field="metadata.related_ids",  # Field to traverse from
        connect_to_field="metadata.document_id",  # Field to match against
        as_field="related_documents",  # Output array field name
        max_depth=2,  # Maximum recursion depth
        depth_field="connection_depth",  # Store depth information
        restrict_search_with_match={  # Additional filters
            "metadata.status": "published",
            "metadata.quality_score": {"$gte": 0.8}
        }
    ),
    enable_graph_enhancement=True
)
```

#### GraphLookupConfig Options

The `GraphLookupConfig` class provides fine-grained control over MongoDB's `$graphLookup` operator for traversing document relationships.


| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `from_collection` | Optional[str] | Target collection for lookup. `None` uses same collection | None |
| `start_with` | str | Expression for starting values (e.g., "$metadata.field") | "$metadata.related_ids" |
| `connect_from_field` | str | Field to traverse from in each iteration | "metadata.related_ids" |
| `connect_to_field` | str | Field to match against | "metadata.document_id" |
| `as_field` | str | Name of output array field containing traversed documents | "graph_connections" |
| `max_depth` | int | Maximum recursion depth (0 = non-recursive) | 2 |
| `depth_field` | Optional[str] | Field to store depth information | "graph_depth" |
| `restrict_search_with_match` | Optional[Dict] | MongoDB query to filter documents during traversal | None |
 -->


## How to run the repo

1. **Install dependencies and provider** (see above)
2. **Configure your MongoDB connection** (Atlas or self-hosted)
3. **Set up your provider in Llama Stack** (see build.yaml or external_providers_dir)
4. **Build and run Llama Stack**:
    ```bash
    llama stack build
    llama stack run
    ```
5. **Use the provider in your code** (see examples below)

---

## Usage examples

### 1. Basic Vector Search

```python
import asyncio
from llama_stack.providers.remote.vector_io.mongodb import MongoDBIOAdapter, MongoDBIOConfig
from llama_stack.apis.vector_dbs import VectorDB

async def vector_search_example():
    # Initialize adapter
    config = MongoDBIOConfig(
        connection_str="mongodb+srv://...",
        namespace="mydb.documents",
        search_mode="vector"
    )
    
    adapter = MongoDBIOAdapter(config, inference_api)
    await adapter.initialize()
    
    # Register a vector database
    vector_db = VectorDB(
        identifier="my_vector_db",
        embedding_dimension=1536,  # OpenAI embeddings dimension
        embedding_type="dense"
    )
    await adapter.register_vector_db(vector_db)
    
    # Insert documents with embeddings
    chunks = [
        Chunk(
            content="The quick brown fox jumps over the lazy dog",
            metadata={"document_id": "doc1", "category": "animals"}
        ),
        Chunk(
            content="Machine learning is a subset of artificial intelligence",
            metadata={"document_id": "doc2", "category": "technology"}
        )
    ]
    
    await adapter.insert_chunks("my_vector_db", chunks)
    
    # Query
    query = "artificial intelligence and ML"
    results = await adapter.query_chunks(
        "my_vector_db",
        query,
    params={"max_chunks": 5, "score_threshold": 0.7}
    )
    
    for chunk, score in zip(results.chunks, results.scores):
        print(f"Score: {score:.3f} - {chunk.content[:100]}")
    
    await adapter.shutdown()

asyncio.run(vector_search_example())
```

### 2. Hybrid Search with Custom Pipelines

```python
async def hybrid_search_example():
    # Configure with custom pipelines for MongoDB 8.1+
    config = MongoDBIOConfig(
        connection_str="mongodb+srv://...",
        namespace="mydb.articles",
        search_mode="native_rank_fusion",
        rank_fusion_pipelines=[
            {
                "name": "semantic_search",
                "type": "vectorSearch",
                "weight": 2.0,  # Give more weight to semantic search
                "limit": 30,
                "config": {
                    "numCandidates": 200,
                    "index": "content_embeddings_index"
                }
            },
            {
                "name": "title_search",
                "type": "search",
                "weight": 1.5,
                "limit": 20,
                "config": {
                    "index": "title_text_index",
                    "operator": "phrase"
                }
            },
            {
                "name": "keyword_match",
                "type": "match",
                "weight": 1.0,
                "limit": 10,
                "config": {
                    "query": {"tags": {"$in": ["AI", "ML", "NLP"]}}
                }
            }
        ],
        enable_score_details=True
    )
    
    adapter = MongoDBIOAdapter(config, inference_api)
    await adapter.initialize()
    
    # Query with both text and vector
    results = await adapter.query_chunks(
        "my_vector_db",
        "Latest developments in natural language processing",
        params={
            "max_chunks": 10,
            "mode": "hybrid",
            "filters": {"metadata.year": {"$gte": 2023}}
            # optional: "ranker": {"strategy": "weighted", "params": {"weights": [0.6, 0.4]}}
        }
    )
    
    # Access detailed scoring information
    for chunk, score in zip(results.chunks, results.scores):
        print(f"Combined Score: {score:.3f}")
        print(f"Title: {chunk.metadata.get('title')}")
        print(f"Content: {chunk.content[:200]}...")
        print("---")
```

### 3. Graph-Enhanced Document Discovery

```python
async def graph_search_example():
    # Configure with graph traversal
    config = MongoDBIOConfig(
        connection_str="mongodb+srv://...",
        namespace="mydb.knowledge_graph",
        search_mode="hybrid_graph",
        graph_lookup_config={
            "connect_from_field": "metadata.references",
            "connect_to_field": "metadata.doc_id", 
            "max_depth": 3,
            "as_field": "related_docs",
            "depth_field": "connection_depth",
            "restrict_search_with_match": {
                "metadata.quality_score": {"$gte": 0.8}
            }
        },
        enable_graph_enhancement=True
    )
    
    adapter = MongoDBIOAdapter(config, inference_api)
    await adapter.initialize()
    
    # Insert documents with relationships
    chunks = [
        Chunk(
            content="Introduction to Neural Networks",
            metadata={
                "doc_id": "nn_intro",
                "references": ["dl_basics", "backprop"],
                "quality_score": 0.95
            }
        ),
        Chunk(
            content="Deep Learning Fundamentals",
            metadata={
                "doc_id": "dl_basics",
                "references": ["ml_intro", "nn_intro"],
                "quality_score": 0.90
            }
        ),
        Chunk(
            content="Backpropagation Algorithm Explained",
            metadata={
                "doc_id": "backprop",
                "references": ["calculus", "nn_intro"],
                "quality_score": 0.85
            }
        )
    ]
    
    await adapter.insert_chunks("my_vector_db", chunks)
    
    # Query will find direct matches and related documents
    results = await adapter.query_chunks(
        "my_vector_db",
        "neural network training",
    params={"max_chunks": 10}
    )
    
    # Results include documents found through graph traversal
    for chunk in results.chunks:
        depth = chunk.metadata.get('connection_depth', 0)
        print(f"Depth {depth}: {chunk.metadata['doc_id']} - {chunk.content[:100]}")
```

### 4. Document Ingestion with Metadata

```python
async def ingest_documents():
    config = MongoDBVectorIOConfig(
        connection_str="mongodb+srv://...",
        namespace="mydb.documents",
        search_mode="native_rank_fusion",
        text_search_fields=["title", "content", "summary", "tags"]
    )
    
    adapter = MongoDBIOAdapter(config, inference_api)
    await adapter.initialize()
    
    # Prepare documents with rich metadata
    documents = [
        {
            "title": "Introduction to Transformers",
            "content": "Transformers are a type of neural network architecture...",
            "summary": "Overview of transformer models in NLP",
            "tags": ["AI", "NLP", "deep learning"],
            "metadata": {
                "document_id": "doc_001",
                "author": "John Doe",
                "date": "2024-01-15",
                "category": "research",
                "related_ids": ["doc_002", "doc_003"],
                "version": "1.0"
            }
        },
        # More documents...
    ]
    
    chunks = []
    for doc in documents:
        chunk = Chunk(
            content=doc["content"],
            metadata={
                **doc["metadata"],
                "title": doc["title"],
                "summary": doc["summary"],
                "tags": doc["tags"]
            }
        )
        chunks.append(chunk)
    
    # Insert with embeddings
    await adapter.insert_chunks("my_vector_db", chunks)
    
    print(f"Ingested {len(chunks)} documents")
```

## Index management

### Creating Indexes Manually

```javascript
// Connect to MongoDB
use mydb

// Create vector search index
db.mycollection.createSearchIndex(
  "vector_index",
  "vectorSearch",
  {
    "fields": [
      {
        "type": "vector",
        "path": "embeddings",
        "numDimensions": 1536,
        "similarity": "cosine"
      }
    ]
  }
)

// Create text search index
db.mycollection.createSearchIndex(
  "text_index",
  "search",
  {
    "mappings": {
      "dynamic": true,
      "fields": {
        "title": {
          "type": "string",
          "analyzer": "lucene.standard"
        },
        "content": {
          "type": "string",
          "analyzer": "lucene.standard"
        }
      }
    }
  }
)

// Create compound index for filtering
db.mycollection.createIndex({
  "metadata.category": 1,
  "metadata.date": -1
})
```

## Performance tips

### 1. Index Optimization

```python
config = MongoDBVectorIOConfig(
    # ... other config ...
    rank_fusion_pipelines=[
        {
            "name": "vector_pipeline",
            "type": "vectorSearch",
            "config": {
                "numCandidates": 500,  # Increase for better recall
                "index": "vector_index"
            }
        }
    ]
)
```

### 2. Batch Processing

```python
async def batch_ingest(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        chunks = [create_chunk(doc) for doc in batch]
        embeddings = await generate_embeddings(chunks)
        await adapter.insert_chunks("my_vector_db", chunks)
        print(f"Processed batch {i//batch_size + 1}")
```

### 3. Connection Pooling

```python
config = MongoDBVectorIOConfig(
    connection_str="mongodb+srv://...?maxPoolSize=50&minPoolSize=10",
    # ... other config ...
)
```

## Monitoring and debugging

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('llama_stack.providers.remote.vector_io.mongodb')
logger.setLevel(logging.DEBUG)
```

### Score Details Analysis

```python
config = MongoDBVectorIOConfig(
    # ... other config ...
    enable_score_details=True
)

# In your query results
for chunk, score in zip(results.chunks, results.scores):
    if hasattr(chunk, 'scoreDetails'):
        print(f"Score breakdown: {chunk.scoreDetails}")
```

## Troubleshooting

### Common Issues

1. **Connection Issues**
   ```python
   # Check connection
   from pymongo import MongoClient
   client = MongoClient(connection_str)
   client.admin.command('ping')
   ```

2. **Index Not Found**
   ```python
   # Verify indexes exist
   db.mycollection.list_search_indexes()
   ```

3. **Version Compatibility**
   ```python
   # Check MongoDB version
   server_info = client.server_info()
   print(f"MongoDB version: {server_info['version']}")
   ```

### Error Messages

| Error | Solution |
|-------|----------|
| `Native rank fusion failed` | MongoDB version < 8.1, will use fallback |
| `Vector DB not found` | Ensure vector DB is registered first |
| `Text query required` | Provide text for hybrid/text search modes |
| `Index not queryable` | Wait for index to be ready (Atlas) |

## Best practices

1. **Choose the Right Search Mode**
   - Use `vector` for semantic similarity
   - Use `full_text` for exact keyword matching
   - Use `hybrid` for balanced results
   - Use `native_rank_fusion` for MongoDB 8.1+ with custom pipelines

2. **Optimize Embeddings**
   - Use appropriate embedding dimensions
   - Consider model-specific requirements
   - Normalize embeddings if needed

3. **Metadata Design**
   - Include filterable fields in metadata
   - Use consistent field names
   - Index frequently queried fields

4. **Production Deployment**
   - Use connection pooling
   - Implement retry logic
   - Monitor query performance
   - Set up proper logging