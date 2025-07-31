from typing import List, Dict, Any, Optional
import uuid
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
import os

from ..config import settings


class VectorStoreService:
    def __init__(self):
        """Initialize the vector store with Azure OpenAI embeddings and ChromaDB"""
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_API_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            model=settings.AZURE_OPENAI_EMBEDDING_MODEL,
        )
        
        # Create persist directory if it doesn't exist
        os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize ChromaDB vector store
        self.vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        
        # Initialize text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_article_content(self, article: Dict[str, Any]) -> List[Document]:
        """
        Chunk article content into smaller pieces for better retrieval
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            List of Document objects with chunked content and metadata
        """
        documents = []
        
        # Combine title and abstract for primary content
        primary_content = f"Title: {article.get('title', '')}\n\nAbstract: {article.get('abstract', '')}"
        
        # Add analysis components if available
        analysis_fields = [
            'objective', 'methodology', 'key_variables', 'main_findings', 
            'implications', 'limitations'
        ]
        
        for field in analysis_fields:
            if article.get(field):
                primary_content += f"\n\n{field.replace('_', ' ').title()}: {article[field]}"
        
        # Create metadata for all chunks
        metadata = {
            "title": article.get('title', ''),
            "authors": ', '.join(article.get('authors', [])),
            "year": article.get('year'),
            "journal": article.get('journal', ''),
            "risk_type": article.get('risk_type', ''),
            "level_of_analysis": article.get('level_of_analysis', ''),
            "source": "crossref",
            "document_id": str(uuid.uuid4())
        }
        
        # Chunk the content
        chunks = self.text_splitter.split_text(primary_content)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
            
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def add_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Add articles to the vector store
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary with statistics about added documents
        """
        all_documents = []
        
        for article in articles:
            if article.get('abstract'):  # Only process articles with abstracts
                docs = self.chunk_article_content(article)
                all_documents.extend(docs)
        
        if all_documents:
            # Add documents to vector store
            self.vector_store.add_documents(all_documents)
        
        return {
            "total_articles": len(articles),
            "processed_articles": len([a for a in articles if a.get('abstract')]),
            "total_chunks": len(all_documents)
        }
    
    def search_similar(self, query: str, k: int = 5, **filters) -> List[Document]:
        """
        Search for similar documents using semantic similarity
        
        Args:
            query: Search query
            k: Number of results to return
            **filters: Additional metadata filters
            
        Returns:
            List of relevant Document objects
        """
        # Apply filters if provided
        search_kwargs = {"k": k}
        
        if filters:
            # Build filter dictionary for ChromaDB
            where_clause = {}
            for key, value in filters.items():
                if value is not None:
                    where_clause[key] = value
            
            if where_clause:
                search_kwargs["filter"] = where_clause
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, **search_kwargs)
        return results
    
    def get_retriever(self, k: int = 5, **search_kwargs) -> VectorStoreRetriever:
        """
        Get a retriever object for use in LangChain chains
        
        Args:
            k: Number of documents to retrieve
            **search_kwargs: Additional search parameters
            
        Returns:
            VectorStoreRetriever object
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": k, **search_kwargs}
        )
    
    def delete_collection(self):
        """Delete the entire collection (useful for testing/reset)"""
        self.vector_store.delete_collection()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection"""
        try:
            collection = self.vector_store._collection
            return {
                "total_documents": collection.count(),
                "collection_name": settings.CHROMA_COLLECTION_NAME
            }
        except Exception as e:
            return {"error": str(e), "total_documents": 0}