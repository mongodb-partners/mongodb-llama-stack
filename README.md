# MongoDB Vector IO Provider for Llama Stack

A MongoDB integration for Llama Stack that provides vector search, full-text search, hybrid search, and graph-enhanced retrieval capabilities using native MongoDB Atlas features.

## Features

- **Vector Search**: Semantic similarity search using embedding vectors
- **Full-Text Search**: Keyword-based search with advanced text analysis
- **Hybrid Search**: Combine semantic and keyword search with flexible weighting
- **Graph-Enhanced Retrieval (TBD)**: Discover related documents through graph traversal
- **RankFusion Pipeline**: Native MongoDB 8.1+ feature for optimal result ranking
- **Self-Managed or Atlas**: Works with both MongoDB Atlas and self-hosted deployments
- **Automatic Index Creation**: Optimized index provisioning for Atlas environments
- **Advanced Filtering**: Combine vector search with metadata filters

## Requirements

- Python 3.10+
- MongoDB Atlas cluster (recommended) or MongoDB 8.0+ instance
- Llama Stack 0.2.0+
- pymongo 4.5.0+

---

## Getting Started

You can integrate this provider with Llama Stack using either the external providers directory method (recommended for development) or by installing it as a Python module.

### Option 1: External Providers Directory (Development Mode)

This approach is ideal for development as it doesn't require reinstallation after code changes.

#### 1. Clone and Set Up the Repository
```bash
# Clone the repository
git clone https://github.com/mongodb-partners/mongodb-llama-stack.git
cd mongodb-llama-stack

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and the package in development mode
pip install -r requirements.txt
pip install -e .
```

#### 2. Configure Your MongoDB Connection

You'll need a MongoDB Atlas cluster or a MongoDB 8.0+ instance with the Search and Vector Search capabilities enabled.

```bash
# Set required environment variables
export MONGODB_CONNECTION_STR='mongodb+srv://<username>:<password>@<cluster-address>/'
export MONGODB_NAMESPACE='<database>.<collection>'
export EXTERNAL_PROVIDERS_DIR="$(pwd)/mongodb_llama_stack/providers.d"
```

For production deployments, consider using a secure method to store and retrieve these credentials.

#### 3. Verify the Connection
```bash
# Run the connection test script
python -m mongodb_llama_stack.mongodb.connection_test

# Expected output:
# MongoDB connection successful!
# Server version: 8.x.x
# Available features: vectorSearch, search, rankFusion, etc.
```

#### 4. Add Provider to Your Llama Stack Configuration
Create or update your `run.yaml` file with the MongoDB provider:

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
        # Optional configuration:
        # search_mode: vector | full_text | hybrid | native_rank_fusion | hybrid_graph
        # index_name: default
        # text_index_name: text_index
        # text_search_fields: ["title", "content", "description"]
external_providers_dir: ${env.EXTERNAL_PROVIDERS_DIR:=~/.llama/providers.d}
```

#### 5. Build and Test the Provider

You can use the included build and test script to ensure everything works correctly:

```bash
# Make the script executable
chmod +x scripts/build_and_test.sh

# Run build and test script
./scripts/build_and_test.sh
```

This script will:
- Set up a virtual environment
- Install all required dependencies
- Test the MongoDB connection
- Run unit tests for each search mode (vector, full-text, hybrid, graph-enhanced)
- Run integration tests against a real MongoDB instance
- Generate coverage reports

##### Available Tests

The testing suite includes:

1. **Unit Tests** (`tests/test_mongodb_provider.py`):
   - Test vector search functionality with different configurations
   - Test full-text search with various analyzers and fields
   - Test hybrid search with different weight configurations
   - Test graph-enhanced document discovery
   - Test index management and automatic creation

2. **Integration Tests** (`tests/integration_test.py`):
   - Test end-to-end document ingestion and retrieval
   - Test search accuracy with real vector embeddings
   - Test filtering with metadata
   - Test performance under various load conditions
   - Test server feature detection and fallbacks

#### 6. Build and Run Llama Stack

```bash
# Build Llama Stack with your configuration
llama stack build

# Run the Llama Stack server
llama stack run
```

You can verify the MongoDB provider is working correctly by checking the logs during startup.

## Search Modes and Configuration

This provider supports multiple search modes that can be configured according to your needs:

- **Vector Search** - Pure semantic search using embeddings
- **Full-Text Search** - Keyword-based search
- **Hybrid Search** - Combined vector and text search
- **Native Rank Fusion** - Advanced results ranking (MongoDB 8.1+)
- **Graph-Enhanced Hybrid Search (TBD)** - Hybrid search with related document discovery

For detailed configuration of each mode, see the [Search Modes Documentation](./docs/search_modes.md).

## Example Usage

For a complete working example, check out the [demo script](./examples/demo.py):

```bash
# Set required environment variables
export MONGODB_CONNECTION_STR='mongodb+srv://<username>:<password>@<cluster-address>/'
export MONGODB_NAMESPACE='demo.documents'

# Run the demo
python examples/demo.py
```

This will demonstrate:
- Document ingestion with automatic embedding generation
- Basic search with hybrid mode (vector + text)
- Filtered search using metadata
- Various configuration options

## Complete Build and Test Process

For a comprehensive build and test of all MongoDB Atlas search integrations:

1. **Set Up MongoDB Atlas**
   - Create a cluster with Vector Search and Atlas Search enabled
   - Create a database user with read/write permissions
   - Whitelist your IP address

2. **Clone and Configure**
   ```bash
   git clone https://github.com/mongodb-partners/mongodb-llama-stack.git
   cd mongodb-llama-stack
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Set Environment Variables**
   ```bash
   export MONGODB_CONNECTION_STR='mongodb+srv://<username>:<password>@<cluster>/'
   export MONGODB_NAMESPACE='test.documents'
   export EXTERNAL_PROVIDERS_DIR="$(pwd)/mongodb_llama_stack/providers.d"
   ```

4. **Test Each Search Mode**
   ```bash
   # Vector search test
   python -m mongodb_llama_stack.mongodb.config_validator --mode vector
   
   # Full text search test
   python -m mongodb_llama_stack.mongodb.config_validator --mode full_text
   
   # Hybrid search test
   python -m mongodb_llama_stack.mongodb.config_validator --mode hybrid
   
   # Native rank fusion test (MongoDB 8.1+)
   python -m mongodb_llama_stack.mongodb.config_validator --mode native_rank_fusion
   
   # Graph-enhanced search test
   python -m mongodb_llama_stack.mongodb.config_validator --mode hybrid_graph
   ```

5. **Run Comprehensive Test Suite**
   ```bash
   # Run all unit tests
   pytest -xvs tests/test_mongodb_provider.py
   
   # Run integration tests
   pytest -xvs tests/integration_test.py
   
   # Run specific test types
   pytest -m vector_search
   pytest -m text_search
   pytest -m hybrid_search
   
   # Generate coverage report
   pytest --cov=mongodb_llama_stack tests/
   ```

6. **Build and Run with Llama Stack**
   ```bash
   # Build Llama Stack with MongoDB provider
   llama stack build
   
   # Run Llama Stack server
   llama stack run
   ```

#### Important Notes
- Example configuration: See `mongodb_llama_stack/run.yaml` for a complete working example
- Provider discovery: The file `mongodb_llama_stack/providers.d/remote/vector_io/mongodb.yaml` defines the provider
- Implementation location: `mongodb_llama_stack/mongodb/` contains the core provider code

### Option 2: Module Installation (Production Use)

This approach is recommended for production deployments or when integrating into existing projects.

#### 1. Install the Package
```bash
# Install from PyPI
pip install mongodb-llama-stack

# Or install from your local build
pip install .
```

#### 2. Reference the Module in Your Configuration
Update your `build.yaml` or `run.yaml` to include the MongoDB provider:

```yaml
providers:
  vector_io:
    - provider_type: remote::mongodb
      module: mongodb_llama_stack
      config:
        connection_str: ${env.MONGODB_CONNECTION_STR:+}
        namespace: ${env.MONGODB_NAMESPACE:+}
        # Additional config options same as Option 1
```

#### 3. Build and Run Llama Stack
```bash
llama stack build
llama stack run
```

> **Note:** The provider is discovered as `remote::mongodb` in both integration methods.

---

## Building and Testing

This section covers how to build, test, and validate the MongoDB provider functionality.

### Building the Provider

```bash
# Clone the repository (if not done already)
git clone https://github.com/mongodb-partners/mongodb-llama-stack.git
cd mongodb-llama-stack

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Run the build script
./scripts/build_and_test.sh --build-only
```

### Running Tests

The repository includes unit tests and integration tests to verify functionality:

```bash
# Run all tests (requires MongoDB connection)
./scripts/build_and_test.sh

# Run only unit tests (no MongoDB connection needed)
./scripts/build_and_test.sh --unit

# Run integration tests (requires MongoDB connection)
./scripts/build_and_test.sh --integration
```

### Testing Specific Features

Test individual search capabilities:

#### Vector Search
```bash
export MONGODB_CONNECTION_STR="mongodb+srv://<user>:<password>@<host>/"
export MONGODB_NAMESPACE="test_llama.vector_test"
python examples/demo.py
```

#### Text Search and Hybrid Search
Run the demo script with specific search modes:

```bash
# Test text search
export MONGODB_SEARCH_MODE="full_text"
python examples/demo.py

# Test hybrid search 
export MONGODB_SEARCH_MODE="hybrid"
python examples/demo.py
```

---


## MongoDB Environment Setup

### MongoDB Atlas (Recommended)

Atlas is the fully managed cloud database service that provides the best experience for this provider.

1. Create a [MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register)
2. Create a new cluster (M0 free tier works for testing)
3. Create a database user with read/write privileges
4. Whitelist your IP address in the Network Access settings
5. Get your connection string from the Atlas Dashboard:
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Select the appropriate driver version
   - Copy the provided connection string

### Self-hosted MongoDB

If using a self-hosted MongoDB instance, ensure you're running MongoDB 8.0+ for full feature support:

```bash
# Check MongoDB version
mongosh --eval "db.version()"
```

For full functionality, we recommend:
- MongoDB 8.0+ for basic vector search
- MongoDB 8.1+ for native rank fusion and advanced hybrid search
- Proper index configuration for your collections

---


## Provider Configuration

### Basic Configuration

The MongoDB provider can be configured programmatically or through environment variables. Here's how to set it up:

#### Programmatic Configuration
```python
from mongodb_llama_stack.mongodb.config import MongoDBIOConfig

# Create configuration object with minimum required settings
config = MongoDBIOConfig(
    connection_str="mongodb+srv://username:password@cluster.mongodb.net/",
    namespace="mydb.mycollection"
)

# Optional: Add adapter configuration
from mongodb_llama_stack.mongodb.mongodb import MongoDBIOAdapter
from llama_stack.apis.inference import InferenceAPI

adapter = MongoDBIOAdapter(config, inference_api)
await adapter.initialize()
```

### Environment Variables

Store your configuration in environment variables or a `.env` file:

```bash
# Required settings
MONGODB_CONNECTION_STR=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_NAMESPACE=mydb.mycollection

# Optional settings
MONGODB_SEARCH_MODE=hybrid
MONGODB_INDEX_NAME=vector_index
MONGODB_TEXT_INDEX_NAME=text_index
MONGODB_EMBEDDINGS_KEY=embeddings
```

### Validation and Troubleshooting

You can validate your configuration using the included test script:

```bash
# Run configuration validation
python -m mongodb_llama_stack.tests.config_validator

# Check MongoDB connection
python -m mongodb_llama_stack.tests.connection_test
```

---


## Search Modes

The MongoDB provider offers multiple search modes to fit different use cases. Choose the best mode based on your retrieval needs.

### 1. Vector Search

**Best for:** Semantic similarity search using embeddings

Vector search excels at finding conceptually similar content even when exact keywords don't match, making it ideal for semantic retrieval tasks.

```python
from mongodb_llama_stack.mongodb.config import MongoDBIOConfig, SearchMode

config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode=SearchMode.VECTOR,
    embeddings_key="embeddings",  # Field containing vector embeddings
    index_name="vector_index"
)
```

**YAML Configuration:**
```yaml
provider_id: mongodb
provider_type: remote::mongodb
config:
  connection_str: ${env.MONGODB_CONNECTION_STR:+}
  namespace: ${env.MONGODB_NAMESPACE:+}
  search_mode: vector
  embeddings_key: embeddings
  index_name: vector_index
```

### 2. Full-Text Search

**Best for:** Keyword-based search with exact matches

Full-text search is optimal for finding documents containing specific words, phrases, or terms, with advanced text analysis capabilities.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode=SearchMode.FULL_TEXT,
    text_index_name="text_index",
    text_search_fields=["title", "content", "description"]
)
```

**YAML Configuration:**
```yaml
provider_id: mongodb
provider_type: remote::mongodb
config:
  connection_str: ${env.MONGODB_CONNECTION_STR:+}
  namespace: ${env.MONGODB_NAMESPACE:+}
  search_mode: full_text
  text_index_name: text_index
  text_search_fields: ["title", "content", "description"]
```

### 3. Hybrid Search

**Best for:** Balanced retrieval combining semantic understanding with keyword matching

Hybrid search combines vector similarity with text matching to get the best of both worlds, ideal for robust RAG applications.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode=SearchMode.HYBRID,
    embeddings_key="embeddings",
    text_search_fields=["title", "content"],
    text_index_name="text_index",
    hybrid_alpha=0.7  # 70% weight to vector, 30% to text
)
```

**YAML Configuration:**
```yaml
provider_id: mongodb
provider_type: remote::mongodb
config:
  connection_str: ${env.MONGODB_CONNECTION_STR:+}
  namespace: ${env.MONGODB_NAMESPACE:+}
  search_mode: hybrid
  embeddings_key: embeddings
  text_search_fields: ["title", "content"]
  text_index_name: text_index
  hybrid_alpha: 0.7
```

### 4. Native Rank Fusion (MongoDB 8.1+)

**Best for:** Advanced multi-pipeline search with fine-grained control

Uses MongoDB's native `$rankFusion` operator for optimal performance and precise control over search pipelines.

```python
config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode=SearchMode.NATIVE_RANK_FUSION,
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

**YAML Configuration:**
```yaml
provider_id: mongodb
provider_type: remote::mongodb
config:
  connection_str: ${env.MONGODB_CONNECTION_STR:+}
  namespace: ${env.MONGODB_NAMESPACE:+}
  search_mode: native_rank_fusion
  rank_fusion_pipelines:
    - name: vector_pipeline
      type: vectorSearch
      weight: 1.5
      limit: 20
      config:
        numCandidates: 100
        index: vector_index
    - name: text_pipeline
      type: search
      weight: 1.0
      limit: 20
      config:
        index: text_index
        operator: phrase
  enable_score_details: true
```

<!-- ### 5. Graph-Enhanced Search

**Best for:** Knowledge graph exploration and discovering related content

Graph-Enhanced Search combines vector or hybrid search with graph traversal to discover related documents through specified relationships.

```python
from mongodb_llama_stack.mongodb.config import (
    MongoDBIOConfig,
    GraphLookupConfig,
    SearchMode
)

config = MongoDBIOConfig(
    connection_str="${env.MONGODB_CONNECTION_STR}",
    namespace="${env.MONGODB_NAMESPACE}",
    search_mode=SearchMode.HYBRID_GRAPH,
    
    # Vector search configuration
    embeddings_key="embeddings",
    index_name="vector_index",
    
    # Graph traversal configuration
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

**YAML Configuration:**
```yaml
provider_id: mongodb
provider_type: remote::mongodb
config:
  connection_str: ${env.MONGODB_CONNECTION_STR:+}
  namespace: ${env.MONGODB_NAMESPACE:+}
  search_mode: hybrid_graph
  embeddings_key: embeddings
  index_name: vector_index
  enable_graph_enhancement: true
  graph_lookup_config:
    from_collection: null
    start_with: "$metadata.related_ids"
    connect_from_field: "metadata.related_ids"
    connect_to_field: "metadata.document_id"
    as_field: "related_documents"
    max_depth: 2
    depth_field: "connection_depth"
    restrict_search_with_match:
      "metadata.status": "published"
      "metadata.quality_score": {"$gte": 0.8}
```

#### GraphLookupConfig Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `from_collection` | Optional[str] | Target collection for lookup. `None` uses same collection | None |
| `start_with` | str | Expression for starting values (e.g., "$metadata.field") | "$metadata.related_ids" |
| `connect_from_field` | str | Field to traverse from in each iteration | "metadata.related_ids" |
| `connect_to_field` | str | Field to match against | "metadata.document_id" |
| `as_field` | str | Name of output array field containing traversed documents | "graph_connections" |
| `max_depth` | int | Maximum recursion depth (0 = non-recursive) | 2 |
| `depth_field` | Optional[str] | Field to store depth information | "graph_depth" |
| `restrict_search_with_match` | Optional[Dict] | MongoDB query to filter documents during traversal | None | -->


## Complete Setup and Testing Workflow

This section provides a step-by-step guide to set up, build, test, and run the MongoDB provider with Llama Stack.

### 1. Installation and Setup

```bash
# Clone the repository
git clone https://github.com/mongodb-partners/mongodb-llama-stack.git
cd mongodb-llama-stack

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the provider with development dependencies
pip install -e ".[dev,test]"
```

### 2. Configure MongoDB Connection

```bash
# Set required environment variables
export MONGODB_CONNECTION_STR='mongodb+srv://username:password@cluster.mongodb.net/'
export MONGODB_NAMESPACE='mydb.mycollection'
export EXTERNAL_PROVIDERS_DIR="$(pwd)/mongodb_llama_stack/providers.d"
```

### 3. Validate Installation and Connection

```bash
# Verify configuration and connection
python -m mongodb_llama_stack.tests.connection_test
```

### 4. Run Tests

```bash
# Run basic tests
pytest tests/test_mongodb_provider.py -v

# Run integration tests (requires active MongoDB connection)
python tests/integration_test.py
```

### 5. Try the Demo

```bash
# Run the demo script to see different search modes in action
python examples/demo.py
```

### 6. Build and Run Llama Stack

```bash
# Set up Llama Stack with the provider
export EXTERNAL_PROVIDERS_DIR="$(pwd)/mongodb_llama_stack/providers.d"
llama stack build
llama stack run
```

### 7. Verify Provider Registration

After starting Llama Stack, you can verify that the provider is registered:

```bash
# Check provider registration
curl http://localhost:8321/registry/providers | jq
```

You should see `remote::mongodb` listed in the providers.

### 8. Use in Applications

Now you can use the provider in your applications that interact with Llama Stack:

```bash
# Configure llama-stack-client to use your server
llama-stack-client configure --endpoint http://localhost:8321 --api-key none

# Test vector search using the client
llama-stack-client vector-io insert-chunks \
  --vector-db-id my_vector_db \
  --provider-id mongodb \
  --content "Test document for MongoDB vector search"
```

---

## Advanced Usage Examples

For detailed examples showcasing various usage scenarios of the MongoDB provider with Llama Stack, see the [Examples Documentation](./docs/examples.md).

For a quick-start example, check the [demo script](./examples/demo.py).


---

## Contributing

For information on how to contribute to this project, please see the [contributing guidelines](./docs/contributing.md).

## License

This project is licensed under the **Apache License 2.0**.
Portions of the code are derived from [Meta’s Llama Stack project](https://github.com/llamastack/llama-stack), licensed under the **MIT License**.

See the full [LICENSE](./LICENSE) file for details.
