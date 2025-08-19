# Metric Research Script

This script was created for the [Distributed vector store](https://ai-llm-applications.gitbook.io/llm-simulator/easy/distributedvectorstore).  
It demonstrates setting up a distributed file-based vector store using Ollama and Python, with included tests for core functionality.

**Main features:**
- Add documents with embeddings
- Search for relevant documents
- Save and load chunked files

**Example usage:**
```python
from dist_vector_store import DistributedVectorStore, DocumentWithEmbedding

# Initialize the vector store
vector_store = DistributedVectorStore()

# Add documents
documents = [
    DocumentWithEmbedding(
        id="1",
        page_content="Current weather in Dublin is 18 degrees",
        metadata={"key": "value"},
        embedding=[0.1, 0.2, 0.3]
    )
]
vector_store.add_documents(documents)

# Search for documents
results = vector_store.search("What is the weather in Dublin?")
```

## Setup

1. **Install Ollama:**
    ```sh
    curl -fsSL https://ollama.ai/install.sh | sh
    ```

2. **Download the embedding model:**
    ```sh
    ollama pull nomic-embed-text
    ```

3. **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the tests by command line or use an 'Testing' tab in your IDE:**
    ```sh
    pytest --cov=dist_vector_store --cov-report=term-missing
    ```