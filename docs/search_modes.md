## Testing Different Search Modes

The MongoDB provider supports multiple search modes that combine semantic and keyword search capabilities. This section shows how to build, test, and configure each mode.

### 1. Vector Search Mode

Vector search performs semantic similarity search using embedding vectors.

```yaml
# Configuration in run.yaml
providers:
  vector_io:
    - provider_id: mongodb
      provider_type: remote::mongodb
      config:
        connection_str: ${env.MONGODB_CONNECTION_STR}
        namespace: ${env.MONGODB_NAMESPACE}
        search_mode: vector
```

To test vector search mode:

```bash
# Test vector search mode
python -m mongodb_llama_stack.mongodb.config_validator --mode vector
```

### 2. Full-Text Search Mode

Full-text search performs keyword-based search using MongoDB's text search capabilities.

```yaml
# Configuration in run.yaml
providers:
  vector_io:
    - provider_id: mongodb
      provider_type: remote::mongodb
      config:
        connection_str: ${env.MONGODB_CONNECTION_STR}
        namespace: ${env.MONGODB_NAMESPACE}
        search_mode: full_text
        text_search_fields: ["title", "content", "metadata.description"]
```

To test full-text search mode:

```bash
# Test full-text search mode with a specific query
python -m mongodb_llama_stack.mongodb.config_validator --mode full_text --query "machine learning"
```

### 3. Hybrid Search Mode

Hybrid search combines both vector and text search for better results.

```yaml
# Configuration in run.yaml
providers:
  vector_io:
    - provider_id: mongodb
      provider_type: remote::mongodb
      config:
        connection_str: ${env.MONGODB_CONNECTION_STR}
        namespace: ${env.MONGODB_NAMESPACE}
        search_mode: hybrid
        text_search_fields: ["title", "content"]
        vector_weight: 2.0
        text_weight: 1.0
```

To test hybrid search mode:

```bash
# Test hybrid search mode
python -m mongodb_llama_stack.mongodb.config_validator --mode hybrid --query "quantum computing"
```

### 4. Native Rank Fusion Mode (MongoDB 8.1+)

This mode uses MongoDB's native rank fusion capabilities for optimal results ranking.

```yaml
# Configuration in run.yaml
providers:
  vector_io:
    - provider_id: mongodb
      provider_type: remote::mongodb
      config:
        connection_str: ${env.MONGODB_CONNECTION_STR}
        namespace: ${env.MONGODB_NAMESPACE}
        search_mode: native_rank_fusion
        vector_weight: 2.0
        text_weight: 1.5
```

To test native rank fusion:

```bash
# Test rank fusion (requires MongoDB 8.1+)
python -m mongodb_llama_stack.mongodb.config_validator --mode native_rank_fusion
```

<!-- ### 5. Graph-Enhanced Hybrid Search

This advanced mode combines hybrid search with graph traversal to find related documents.

```yaml
# Configuration in run.yaml
providers:
  vector_io:
    - provider_id: mongodb
      provider_type: remote::mongodb
      config:
        connection_str: ${env.MONGODB_CONNECTION_STR}
        namespace: ${env.MONGODB_NAMESPACE}
        search_mode: hybrid_graph
        graph_max_depth: 2
        related_field: "metadata.related_ids"
```

To test graph-enhanced search:

```bash
# Test graph-enhanced search mode
python -m mongodb_llama_stack.mongodb.config_validator --mode hybrid_graph
``` -->

## Using the Provider in Python Code

You can also use the MongoDB provider directly from your Python code:

```python
import asyncio
from llama_stack.apis.vector_io import Chunk
from llama_stack.apis.inference import InferenceAPI
from mongodb_llama_stack.mongodb.config import MongoDBIOConfig, SearchMode
from mongodb_llama_stack.mongodb.mongodb import MongoDBIOAdapter

# Create a configuration
config = MongoDBIOConfig(
    connection_str="mongodb+srv://username:password@cluster.mongodb.net/",
    namespace="mydb.mycollection",
    search_mode=SearchMode.HYBRID,
    text_search_fields=["content", "metadata.title"],
    vector_weight=2.0,
    text_weight=1.0
)

# Initialize with a real embedding model from Llama Stack
async def main():
    inference_api = InferenceAPI()  # Your actual embedding model
    provider = MongoDBIOAdapter(config=config, inference_api=inference_api)
    
    # Initialize provider
    await provider.initialize()
    
    # Store documents
    chunks = [
        Chunk(
            content="This is a sample document about MongoDB",
            metadata={"title": "MongoDB Introduction", "category": "databases"}
        ),
        # Add more documents...
    ]
    await provider.store(chunks)
    
    # Search documents
    results = await provider.search(
        "MongoDB vector search",
        limit=5,
        metadata_filter={"metadata.category": "databases"}
    )
    
    # Process results
    for chunk in results:
        print(f"Content: {chunk.content}")
        print(f"Score: {chunk.score}")
        print(f"Metadata: {chunk.metadata}")
        print("---")

# Run the async function
asyncio.run(main())
```

## Advanced Configuration Options

The MongoDB provider supports numerous configuration options for fine-tuning:

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `connection_str` | string | MongoDB connection string | (Required) |
| `namespace` | string | Database and collection name in format "db.collection" | (Required) |
| `search_mode` | string | Search mode to use: vector, full_text, hybrid, native_rank_fusion, hybrid_graph | "vector" |
| `index_name` | string | Name of the vector search index | "vector_index" |
| `text_index_name` | string | Name of the text search index | "text_index" |
| `text_search_fields` | list | Fields to search in text search | ["content"] |
| `embedding_field` | string | Field containing embedding vectors | "embeddings" |
| `vector_weight` | float | Weight for vector search in hybrid modes | 1.0 |
| `text_weight` | float | Weight for text search in hybrid modes | 1.0 |
| `create_indexes` | bool | Whether to create missing indexes automatically | true |
| `graph_max_depth` | int | Maximum depth for graph traversal | 1 |
| `related_field` | string | Field containing related document IDs | "related_ids" |
