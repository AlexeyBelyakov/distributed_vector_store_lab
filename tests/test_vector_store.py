import pytest
from dist_vector_store import DistributedVectorStore
from langchain_core.documents import Document

@pytest.fixture
def vector_store():
    store = DistributedVectorStore(max_file_size=1024 * 1024)  # 1 MB for testing
    yield store
    store.delete()

def get_documents(scale: int) -> list[Document]:
    docs = [
        Document(page_content="ML is a rapidly evolving field with many applications.", metadata={"source": "test_source_1"}),
        Document(page_content="Italian cuisine is known for its regional diversity.", metadata={"source": "test_source_2"}),
        Document(page_content="The capital of France is Paris.", metadata={"source": "test_source_3"}),
        Document(page_content="Python is a popular programming language for data science.", metadata={"source": "test_source_4"}),
        Document(page_content="Mount Everest is the highest mountain in the world.", metadata={"source": "test_source_5"}),
        Document(page_content="The Great Wall of China is visible from space.", metadata={"source": "test_source_6"}),
        Document(page_content="Photosynthesis is the process by which plants make food.", metadata={"source": "test_source_7"}),
        Document(page_content="The human brain contains approximately 86 billion neurons.", metadata={"source": "test_source_8"}),
        Document(page_content="Renewable energy sources include solar and wind power.", metadata={"source": "test_source_9"}),
        Document(page_content="The periodic table organizes elements by atomic number.", metadata={"source": "test_source_10"}),
    ]
    for doc in docs:
        doc.page_content *= scale
    return docs

@pytest.fixture
def texts_and_metadatas() -> tuple[list[str], list[dict]]:
    texts = ["Test text 1", "Test text 2"]
    metadata_list = [{"source": "test_source_1"}, {"source": "test_source_2"}]
    return texts, metadata_list

@pytest.mark.parametrize("scale", [1, 10, 1000])
def test_add_documents(vector_store: DistributedVectorStore, scale: int):
    documents = get_documents(scale)
    doc_ids = vector_store.add_documents(documents)
    assert len(doc_ids) == len(documents)
    assert all(isinstance(doc_id, str) for doc_id in doc_ids)

def test_add_texts(vector_store: DistributedVectorStore, texts_and_metadatas):
    texts, metadata_list = texts_and_metadatas
    doc_ids = vector_store.add_texts(texts, metadata_list)
    assert len(doc_ids) == len(texts)
    assert all(isinstance(doc_id, str) for doc_id in doc_ids)
    
    documents = vector_store.get_by_ids(doc_ids)
    assert len(documents) == len(texts)
    for i, doc in enumerate(documents):
        assert doc.page_content == texts[i]
        assert doc.metadata.get("source") == metadata_list[i]["source"]
        
    vector_store.delete(doc_ids)
    
    empty_docs = vector_store.get_by_ids(doc_ids)
    assert len(empty_docs) == 0

@pytest.mark.parametrize("scale", [1, 10, 3000])
def test_search(vector_store: DistributedVectorStore, scale: int):
    documents = get_documents(scale)
    all_ids = vector_store.add_documents(documents)
    results = vector_store.search("Chemistry", k=3)
    for result in results:
        assert isinstance(result, Document)
        print(result.page_content)
        # assert "Chemistry" in result.page_content
    assert len(results) == 3
    
    # Check if all documents can be retrieved
    second_store = DistributedVectorStore()
    all_documents = second_store.get_all_documents()
    all_ids2 = [doc.id for doc in all_documents]
    assert set(all_ids2) == set(all_ids)

    # Clean up
    doc_ids = [doc.id for doc in results]
    vector_store.delete(doc_ids)
    
    vector_store.delete(all_ids)
    chunk_files = vector_store._get_chunk_files()
    
    
    assert len(chunk_files) == 0