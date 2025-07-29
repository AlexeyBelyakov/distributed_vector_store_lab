from typing import Optional
import numpy as np
import ollama
from langchain_core.documents import Document
import os
import json
import pickle
import heapq
from dataclasses import dataclass
import uuid

@dataclass
class DocumentWithEmbedding:
    id: str
    page_content: str
    metadata: dict
    embedding: list[float]

    # implement < 
    def __lt__(self, other: 'DocumentWithEmbedding') -> bool:
        return self.id < other.id

    def get_document(self) -> Document:
        return Document(
            page_content=self.page_content,
            metadata=self.metadata,
            id=self.id
        )
    
    @staticmethod
    def load_from_file(file_path: str) -> list['DocumentWithEmbedding']:
        if not os.path.exists(file_path):
            return []
        try:
            with open(file_path, 'r') as f:
                items = json.load(f)
                return [DocumentWithEmbedding(**item) for item in items]
        except json.JSONDecodeError:
            return []

    @staticmethod
    def dump_documents_to_file(documents: list['DocumentWithEmbedding'], file_path: str):
        with open(file_path, 'w') as f:
            documents_dict = [doc.__dict__ for doc in documents]
            json.dump(documents_dict, f, indent=4)

    def dump_to_file(self, file_path: str):
        documents = self.load_from_file(file_path)
        documents.append(self)
        self.dump_documents_to_file(documents, file_path)

class DistributedVectorStore:
    MAX_FILE_SIZE_IN_BYTES = 1024 * 1024 * 1024
    def __init__(self, max_file_size: int = MAX_FILE_SIZE_IN_BYTES):
        self.max_file_size = max_file_size
        self.chunk_file_template = f"vector_store/chunk_{{idx}}.pkl"
        os.makedirs("vector_store", exist_ok=True)
        chunk_files = self._get_chunk_files()
        if chunk_files:
            self.current_chunk_index = max(int(f.split('_')[-1].split('.')[0]) for f in chunk_files)
            self.current_chunk_size = os.path.getsize(chunk_files[self.current_chunk_index])
        else:
            self.current_chunk_index = 0
            self.current_chunk_size = 0

    def add_texts(self, texts: list[str], metadata_list: Optional[list[dict]] = None) -> list[str]:
        documents: list[Document] = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            documents.append(Document(text, metadata=metadata))
        return self.add_documents(documents)

    def delete(self, ids_to_remove: Optional[list[str]] = None) -> None:
        chunk_files_to_update  = self._get_chunk_files()
        for chunk_file in chunk_files_to_update:
            if ids_to_remove is None:
                os.remove(chunk_file)
                continue
            documents = DocumentWithEmbedding.load_from_file(chunk_file)
            documents = [doc for doc in documents if doc.id not in ids_to_remove]
            if documents:
                DocumentWithEmbedding.dump_documents_to_file(documents, chunk_file)
            else:
                os.remove(chunk_file)

    def get_by_ids(self, ids: list[str] = None) -> list[Document]:
        chunk_files = self._get_chunk_files()
        documents = []
        for chunk_file in chunk_files:
            chunk_documents = DocumentWithEmbedding.load_from_file(chunk_file)
            for doc in chunk_documents:
                if ids is None or doc.id in ids:
                    documents.append(doc.get_document())
        return documents
    
    def get_all_documents(self) -> list[Document]:
        return self.get_by_ids()

    def search(self, query: str, k: int) -> list[Document]:
        query_embedding = self._text_to_vec(query)
        heap = []
        chunk_files = self._get_chunk_files()
        for chunk_file in chunk_files:
            docs = DocumentWithEmbedding.load_from_file(chunk_file)
            for doc in docs:
                similarity = self._cosin_similarity(query_embedding, np.array(doc.embedding))
                if len(heap) < k:
                    heapq.heappush(heap, (similarity, doc))
                elif similarity > heap[0][0]:  # если similarity больше минимального
                    heapq.heapreplace(heap, (similarity, doc))

        result = [doc for similarity, doc in sorted(heap, reverse=True)]
        doc_result = [doc.get_document() for doc in result]
        return doc_result
    
    def _prepare_document_with_emb(self, document: Document) -> DocumentWithEmbedding:
        if not hasattr(document, 'id') or not document.id:
            document.id = str(uuid.uuid4())
        embedding = list(self._text_to_vec(document.page_content))
        return DocumentWithEmbedding(
            id=document.id,
            page_content=document.page_content,
            metadata=document.metadata,
            embedding=embedding
        )
        
    def _save_to_chunk_file(self, doc_emb: DocumentWithEmbedding) -> None:
        estimated_size = len(doc_emb.page_content.encode('utf-8')) + 1000
        if self.current_chunk_size + estimated_size > self.max_file_size:
            self.current_chunk_index += 1
            self.current_chunk_size = 0
        self.current_chunk_size += estimated_size
        current_chunk_file = self.chunk_file_template.format(idx=self.current_chunk_index)
        doc_emb.dump_to_file(current_chunk_file)

    def add_documents(self, documents: list[Document]) -> list[str]:
        for document in documents:
            doc_emb = self._prepare_document_with_emb(document)
            self._save_to_chunk_file(doc_emb)
        ids = [doc.id for doc in documents]
        return ids

    def _cosin_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _get_chunk_files(self) -> list[str]:
        dir_name = "vector_store"
        return [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.startswith("chunk_") and f.endswith(".pkl")]

    
    @staticmethod
    def _text_to_vec(text: str) -> np.ndarray:
        try:
            response = ollama.embeddings(model="nomic-embed-text", prompt=text)
            return np.array(response.embedding)
        except Exception as e:
            print(f"Error occurred while embedding text: {e}")
            return np.array([])
