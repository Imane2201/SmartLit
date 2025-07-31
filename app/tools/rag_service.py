from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from .vector_store import VectorStoreService
from ..config import settings


class RAGService:
    def __init__(self):
        """Initialize the RAG service with vector store and Azure OpenAI"""
        self.vector_store = VectorStoreService()
        
        # Initialize Azure OpenAI chat model
        self.llm = AzureChatOpenAI(
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_API_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            model=settings.AZURE_OPENAI_MODEL_NAME,
            temperature=0.3,
        )
        
        # Define the QA prompt template
        self.qa_prompt = PromptTemplate.from_template("""
You are an expert research assistant specializing in academic literature analysis. Use the following pieces of retrieved context from research articles to answer the question. 

Guidelines:
1. Base your answer strictly on the provided context
2. If the context doesn't contain enough information, clearly state what's missing
3. Cite specific findings, methodologies, or insights from the articles when relevant
4. Maintain academic rigor in your response
5. If multiple articles discuss the topic, synthesize the information and note any contradictions

Context from research articles:
{context}

Question: {input}

Provide a comprehensive, well-structured answer based on the retrieved research:
""")
    
    async def query_knowledge_base(
        self, 
        question: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge base using RAG
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            filters: Optional filters for retrieval
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Get retriever with optional filters
        retriever = self.vector_store.get_retriever(k=k)
        
        if filters:
            # Apply filters to the retriever
            search_kwargs = retriever.search_kwargs.copy()
            search_kwargs.update({"filter": filters})
            retriever.search_kwargs = search_kwargs
        
        # Create the documents chain
        document_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Execute the chain
        result = await retrieval_chain.ainvoke({"input": question})
        
        # Extract source information
        sources = []
        for doc in result.get("context", []):
            source_info = {
                "title": doc.metadata.get("title", "Unknown"),
                "authors": doc.metadata.get("authors", "Unknown"),
                "year": doc.metadata.get("year"),
                "journal": doc.metadata.get("journal", "Unknown"),
                "risk_type": doc.metadata.get("risk_type"),
                "chunk_content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "question": question,
            "total_sources": len(sources)
        }
    
    async def multi_article_summary(
        self, 
        article_titles: List[str], 
        focus_question: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a synthesized summary across multiple specific articles
        
        Args:
            article_titles: List of article titles to analyze
            focus_question: Optional specific question to focus the summary
            
        Returns:
            Dictionary containing the synthesized summary and analysis
        """
        # Build a query to retrieve documents from specific articles
        title_query = " OR ".join(f'"{title}"' for title in article_titles)
        
        # Retrieve documents from the specified articles
        relevant_docs = []
        for title in article_titles:
            docs = self.vector_store.search_similar(
                query=title,
                k=10,  # Get more chunks per article
                title=title  # Filter by exact title match
            )
            relevant_docs.extend(docs)
        
        if not relevant_docs:
            return {
                "summary": "No articles found with the specified titles.",
                "articles_found": 0,
                "focus_question": focus_question
            }
        
        # Create a synthesis prompt
        synthesis_prompt = PromptTemplate.from_template("""
You are tasked with synthesizing insights from multiple research articles. Analyze the following content from {num_articles} research articles and provide a comprehensive synthesis.

{focus_instruction}

Content from articles:
{context}

Provide a structured synthesis that includes:
1. **Common Themes**: What consistent patterns or themes emerge across the articles?
2. **Methodological Approaches**: What research methods were used and how do they compare?
3. **Key Findings**: What are the main findings and how do they relate to each other?
4. **Contradictions or Gaps**: Are there any conflicting findings or notable gaps?
5. **Implications**: What are the broader implications of these combined findings?
6. **Future Research Directions**: What areas need further investigation?

Synthesis:
""")
        
        # Prepare the context
        context = "\n\n".join([
            f"Article: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
            for doc in relevant_docs
        ])
        
        focus_instruction = (
            f"Focus your synthesis specifically on: {focus_question}"
            if focus_question
            else "Provide a general synthesis of the key insights."
        )
        
        # Generate the synthesis
        chain = synthesis_prompt | self.llm | StrOutputParser()
        
        synthesis = await chain.ainvoke({
            "context": context,
            "num_articles": len(set(doc.metadata.get('title') for doc in relevant_docs)),
            "focus_instruction": focus_instruction
        })
        
        # Extract unique articles found
        unique_articles = {}
        for doc in relevant_docs:
            title = doc.metadata.get('title', 'Unknown')
            if title not in unique_articles:
                unique_articles[title] = {
                    "title": title,
                    "authors": doc.metadata.get('authors', 'Unknown'),
                    "year": doc.metadata.get('year'),
                    "journal": doc.metadata.get('journal', 'Unknown')
                }
        
        return {
            "summary": synthesis,
            "articles_analyzed": list(unique_articles.values()),
            "articles_found": len(unique_articles),
            "focus_question": focus_question,
            "total_chunks_analyzed": len(relevant_docs)
        }
    
    async def suggest_research_gaps(self, domain: str = "risk management") -> Dict[str, Any]:
        """
        Analyze the knowledge base to suggest research gaps
        
        Args:
            domain: Research domain to focus on
            
        Returns:
            Dictionary containing suggested research gaps and analysis
        """
        # Query for recent research in the domain
        recent_docs = self.vector_store.search_similar(
            query=f"{domain} research methodology findings",
            k=20
        )
        
        if not recent_docs:
            return {
                "gaps": ["No articles found in the knowledge base for analysis."],
                "domain": domain
            }
        
        gap_analysis_prompt = PromptTemplate.from_template("""
You are a senior research analyst tasked with identifying research gaps in {domain}. 

Analyze the following research content and identify:
1. **Underexplored Areas**: What topics or subtopics appear to be missing or underrepresented?
2. **Methodological Gaps**: Are there research methods that could be applied but haven't been used?
3. **Geographic/Temporal Gaps**: Are there regions, time periods, or contexts that lack coverage?
4. **Interdisciplinary Opportunities**: Where could interdisciplinary approaches add value?
5. **Practical Application Gaps**: What bridges between theory and practice are missing?

Research Content:
{context}

Based on this analysis, provide 5-7 specific, actionable research gap suggestions. For each gap, explain:
- What is missing
- Why it's important
- How it could be addressed

Research Gap Analysis:
""")
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Study: {doc.metadata.get('title', 'Unknown')}\n"
            f"Focus: {doc.metadata.get('risk_type', 'General')}\n"
            f"Method: {doc.page_content[:300]}..."
            for doc in recent_docs
        ])
        
        # Generate gap analysis
        chain = gap_analysis_prompt | self.llm | StrOutputParser()
        
        analysis = await chain.ainvoke({
            "context": context,
            "domain": domain
        })
        
        return {
            "gap_analysis": analysis,
            "domain": domain,
            "articles_analyzed": len(set(doc.metadata.get('title') for doc in recent_docs)),
            "coverage_areas": list(set(doc.metadata.get('risk_type') for doc in recent_docs if doc.metadata.get('risk_type')))
        }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the current knowledge base"""
        return self.vector_store.get_collection_stats()