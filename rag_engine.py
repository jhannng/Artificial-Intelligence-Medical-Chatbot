# Standard Libraries
from dotenv import load_dotenv
from typing import List
import os

# Document Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import glob

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter                                     # Split the whole document which containing all text into chunks

# OpenAI API Services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI                                            

# Vector Stores
from langchain_community.vectorstores import Chroma                                                     # Use to store, index, and search through large collection of vector embeddings efficiently

# Retrieval-Augmented Generation (RAG) Chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.prompts import ChatPromptTemplate

# API Key Configuration
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RetrievalAugmentedGenerationEngine:
    # Database Configurations
    DATASET_PATH = "./data/"
    VECTOR_DATABASE_PATH = "./vector_database/chroma_db"
    
    def __init__(self, large_language_model_name: str = "gpt-4o-mini", embedding_model_name: str = "text-embedding-3-small", embedding_chunk_size: int = 500) -> None:
        self.docs_loaders = {"*.pdf": PyPDFLoader, "*.docx": Docx2txtLoader, "*.txt": TextLoader}
        
        # Embedding Model
        self.embedding_model = OpenAIEmbeddings(model = embedding_model_name, openai_api_key = OPENAI_API_KEY)
        
        # Large Language Model
        self.large_language_model = ChatOpenAI(model = large_language_model_name, temperature = 0.3, max_tokens = 512, openai_api_key = OPENAI_API_KEY)
        
        # Retrieval-Augmented Generation (RAG) Chain
        self.chain = None
        
    def load_documents(self) -> List:
        docs = []
        
        for file_type, loader_cls in self.docs_loaders.items():
            file_paths = glob.glob(f"{self.DATASET_PATH}/{file_type}")
            
            for file_path in file_paths:
                try:
                    docs.extend(loader_cls(file_path).load())
                
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                
        return docs
    
    def split_documents(self, documents: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        chunks = text_splitter.split_documents(self.load_documents())
        
        # print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
        
        return chunks
    
    def build_vector_database(self, chunks: List) -> Chroma:
        vector_database = Chroma.from_documents(documents = chunks, embedding = self.embedding_model, persist_directory = self.VECTOR_DATABASE_PATH)
        
        # print(f"Vector Database built successfully.")
        
        return vector_database
    
    def save_vector_database(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        # Load Documents
        documents = self.load_documents()
        
        if not documents:
            print("No documents found to process!")
            return
        
        # Split Documents into Chunks
        chunks = self.split_documents(documents, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        
        # Build Vector Database
        self.build_vector_database(chunks)
        
        # print(f"Vector Database saved to {self.VECTOR_DATABASE_PATH} | Total Chunk Indexed: ({len(chunks)}")
    
    def load_vector_database(self) -> Chroma:
        try:
            vector_database = Chroma(persist_directory = self.VECTOR_DATABASE_PATH, embedding_function = self.embedding_model)
            
            # print(f"Vector Database loaded successfully from {self.VECTOR_DATABASE_PATH}.")
            
            return vector_database
        
        except Exception as e:
            print(f"Error loading vector database: {str(e)}")
            print("You may need to create the database first by calling save_vector_database()")
            raise
    
    def build_rag_chain(self, k: int = 3) -> None:
        # Load Vector Database
        database = self.load_vector_database()
        
        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a knowledgeable medical assisstant and use ONLY the provided context to answer the question. If the answer is not contained in the context, say 'I do not have enough information to answer this question.'\n\n{context}"),
            ("human", "Question:\n{input}\n\nContext:\n{context}")
        ])
        
        doc_chain = create_stuff_documents_chain(llm = self.large_language_model, prompt = prompt)
        
        # Build a Retrieval-Augmented Generation (RAG) Chain
        self.chain = create_retrieval_chain(database.as_retriever(search_kwargs = {"k": k}), combine_docs_chain = doc_chain)
        
        # print(f"Retrieval-Augmented Generation (RAG) Chains initialised.")
    
    def answer_query(self, query: str) -> str:
        if not self.chain:
            raise RuntimeError("RAG chain not initialized. Call build_rag_chain() first.")
        
        # print("\n--- Processing Query ---")
        
        return self.chain.invoke({"input": query})
    
    def similarity_search(self, query: str, k: int = 3) -> List:
        database = self.load_vector_database()
        
        return database.similarity_search(query, k = k)
    
    def update_model(self, large_language_model_name: str = None) -> None:
        self.large_language_model = ChatOpenAI(model = large_language_model_name, temperature = 0.3, max_tokens = 512, openai_api_key = OPENAI_API_KEY)
        
        if self.chain:
            # Rebuild RAG Chain with the new model
            self.build_rag_chain()
        
        # print("Models updated successfully.")