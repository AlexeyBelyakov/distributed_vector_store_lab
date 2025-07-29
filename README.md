# Metric Research Script

This script was prepared for the [LLM Usage Course](https://ai-llm-applications.gitbook.io/llm-simulator/easy/metrics-research).

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