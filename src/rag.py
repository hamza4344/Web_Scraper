
import logging
import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH

class RAGProcessor:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.setup_embeddings()
    
    def setup_embeddings(self):
        """Initialize embedding model"""
        try:
            logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logging.info("Embedding model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
    
    def create_vector_store(self, documents):
        """Create vector store from documents"""
        try:
            logging.info(f"Creating vector store from {len(documents)} documents...")
            
            # Filter documents with meaningful content
            valid_docs = [doc for doc in documents if len(doc.page_content.strip()) > 30]
            
            if not valid_docs:
                logging.error("No valid documents for vector store creation")
                return None
            
            self.vector_store = FAISS.from_documents(valid_docs, self.embeddings)
            logging.info(f"Vector store created with {len(valid_docs)} documents")
            return self.vector_store
            
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            return None
    
    def save_vector_store(self, path=None):
        """Save vector store to disk"""
        if not self.vector_store:
            logging.error("No vector store to save")
            return False
            
        save_path = path or VECTOR_STORE_PATH
        try:
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save_local(save_path)
            logging.info(f"Vector store saved to {save_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save vector store: {e}")
            return False
    
    def load_vector_store(self, path=None):
        """Load vector store from disk"""
        load_path = path or VECTOR_STORE_PATH
        try:
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logging.info(f"Vector store loaded from {load_path}")
            return self.vector_store
        except Exception as e:
            logging.error(f"Failed to load vector store from {load_path}: {e}")
            return None
    
    def similarity_search(self, query, k=5):
        """Perform similarity search"""
        if not self.vector_store:
            logging.error("Vector store not initialized")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logging.info(f"Found {len(results)} results for query: '{query[:50]}...'")
            return results
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return []
    
    def get_retriever(self, k=5):
        """Get retriever for LangChain integration"""
        if not self.vector_store:
            logging.error("Vector store not initialized")
            return None
        return self.vector_store.as_retriever(search_kwargs={"k": k})
