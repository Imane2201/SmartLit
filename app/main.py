from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from typing import List, Optional
import json
from dotenv import load_dotenv
from pydantic import BaseModel

from app.tools.crossref import CrossRefSearchTool
from app.tools.article_analyzer import ArticleAnalyzer
from app.tools.sheets_handler import GoogleSheetsHandler
from app.tools.vector_store import VectorStoreService
from app.tools.rag_service import RAGService
from app.tools.pdf_processor import PDFProcessor
from app.tools.citation_graph import CitationGraphGenerator
from app.tools.article_monitor import ArticleMonitor, run_manual_monitoring

# Load environment variables
load_dotenv()

# Pydantic models for request bodies
class QueryRequest(BaseModel):
    question: str
    k: int = 5
    filters: Optional[dict] = None

class MultiArticleSummaryRequest(BaseModel):
    article_titles: List[str]
    focus_question: Optional[str] = None

class SearchFilters(BaseModel):
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    risk_type: Optional[str] = None
    journal: Optional[str] = None
    level_of_analysis: Optional[str] = None

app = FastAPI(title="SmartLit API", description="Academic Literature Analysis with RAG", version="2.0.0")
crossref_tool = CrossRefSearchTool()
article_analyzer = ArticleAnalyzer()
sheets_handler = GoogleSheetsHandler()
vector_store = VectorStoreService()
rag_service = RAGService()
pdf_processor = PDFProcessor()
citation_graph = CitationGraphGenerator()
article_monitor = ArticleMonitor()

@app.on_event("startup")
async def startup_event():
    # Initialize the Google Sheet with headers
    sheets_handler.initialize_sheet()
    print("✅ Google Sheets initialized")
    
    # Initialize vector store
    stats = vector_store.get_collection_stats()
    print(f"✅ Vector store initialized with {stats.get('total_documents', 0)} documents")
    
    print("🚀 SmartLit API with RAG is ready!")

# Add a root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

@app.post("/search_articles")
async def search_articles(
    topic: str,
    filters: Optional[SearchFilters] = None
):
    """
    Search for articles and analyze them, storing results in both Google Sheets and vector store
    """
    # Search for articles
    articles = crossref_tool._run(topic)
    
    results = []
    for article in articles:
        if article["abstract"]:
            # Analyze article
            analysis, token_usage = await article_analyzer.analyze(article["abstract"])
            
            # Print token usage
            print(f"Token usage for article '{article['title']}': {json.dumps(token_usage, indent=2)}")
            
            # Combine metadata with analysis
            full_article = {**article, **analysis}
            
            # Apply filters if provided
            if filters:
                if filters.year_from and article.get('year') and article['year'] < filters.year_from:
                    continue
                if filters.year_to and article.get('year') and article['year'] > filters.year_to:
                    continue
                if filters.risk_type and analysis.get('risk_type') != filters.risk_type:
                    continue
                if filters.journal and article.get('journal') != filters.journal:
                    continue
                if filters.level_of_analysis and analysis.get('level_of_analysis') != filters.level_of_analysis:
                    continue
            
            results.append(full_article)
    
    # Store results in Google Sheet
    if results:
        sheets_handler.append_articles(results)
        
        # Add to vector store for RAG
        vector_stats = vector_store.add_articles(results)
        print(f"Added {vector_stats['total_chunks']} chunks to vector store from {vector_stats['processed_articles']} articles")
    
    return {
        "articles": results,
        "total_found": len(results),
        "topic": topic,
        "filters_applied": filters.dict() if filters else None
    }

@app.post("/query_knowledge_base")
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using RAG to answer questions about the research articles
    """
    try:
        result = await rag_service.query_knowledge_base(
            question=request.question,
            k=request.k,
            filters=request.filters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

@app.post("/multi_article_summary")
async def multi_article_summary(request: MultiArticleSummaryRequest):
    """
    Generate a synthesized summary across multiple specific articles
    """
    try:
        result = await rag_service.multi_article_summary(
            article_titles=request.article_titles,
            focus_question=request.focus_question
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating multi-article summary: {str(e)}")

@app.get("/suggest_research_gaps")
async def suggest_research_gaps(domain: str = Query(default="risk management")):
    """
    Analyze the knowledge base to suggest research gaps in a specific domain
    """
    try:
        result = await rag_service.suggest_research_gaps(domain=domain)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing research gaps: {str(e)}")

@app.get("/knowledge_base_stats")
async def get_knowledge_base_stats():
    """
    Get statistics about the current knowledge base
    """
    try:
        stats = rag_service.get_knowledge_base_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting knowledge base stats: {str(e)}")

@app.get("/search_similar")
async def search_similar(
    query: str = Query(..., description="Search query"),
    k: int = Query(default=5, description="Number of results to return"),
    risk_type: Optional[str] = Query(default=None, description="Filter by risk type"),
    year: Optional[int] = Query(default=None, description="Filter by year"),
    journal: Optional[str] = Query(default=None, description="Filter by journal")
):
    """
    Search for similar documents in the vector store
    """
    try:
        filters = {}
        if risk_type:
            filters["risk_type"] = risk_type
        if year:
            filters["year"] = year
        if journal:
            filters["journal"] = journal
        
        results = vector_store.search_similar(query=query, k=k, **filters)
        
        # Format results for API response
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": getattr(doc, 'relevance_score', None)
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results),
            "filters_applied": filters
        }
     except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error searching similar documents: {str(e)}")

@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    authors: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    journal: Optional[str] = Form(None)
):
    """
    Upload and process a PDF research paper
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read the file content
        pdf_content = await file.read()
        
        # Validate the PDF
        validation = pdf_processor.validate_pdf(pdf_content)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["error"])
        
        # Prepare custom metadata if provided
        custom_metadata = {}
        if title:
            custom_metadata["title"] = title
        if authors:
            custom_metadata["authors"] = [author.strip() for author in authors.split(',')]
        if year:
            custom_metadata["year"] = year
        if journal:
            custom_metadata["journal"] = journal
        
        # Process the PDF
        result = await pdf_processor.process_pdf(
            pdf_content=pdf_content,
            filename=file.filename,
            custom_metadata=custom_metadata if custom_metadata else None
        )
        
        if result["success"]:
            # Add to Google Sheets if analysis was successful
            try:
                sheets_handler.append_articles([result["article"]])
                result["added_to_sheets"] = True
            except Exception as e:
                result["added_to_sheets"] = False
                result["sheets_error"] = str(e)
            
            return {
                "success": True,
                "message": "PDF processed successfully",
                "filename": file.filename,
                "article": {
                    "title": result["article"].get("title", ""),
                    "authors": result["article"].get("authors", []),
                    "year": result["article"].get("year"),
                    "journal": result["article"].get("journal", ""),
                    "risk_type": result["article"].get("risk_type", ""),
                    "level_of_analysis": result["article"].get("level_of_analysis", ""),
                    "objective": result["article"].get("objective", "")[:200] + "..." if result["article"].get("objective") else "",
                    "main_findings": result["article"].get("main_findings", "")[:200] + "..." if result["article"].get("main_findings") else ""
                },
                "processing_stats": {
                    "extracted_text_length": result["extracted_text_length"],
                    "chunks_added": result["vector_stats"]["total_chunks"],
                    "token_usage": result["token_usage"],
                    "added_to_sheets": result.get("added_to_sheets", False)
                }
            }
        else:
            raise HTTPException(status_code=400, detail=f"PDF processing failed: {result['error']}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/pdf_info")
async def get_pdf_info(file: UploadFile = File(...)):
    """
    Get basic information about a PDF file without processing it
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        pdf_content = await file.read()
        info = pdf_processor.get_pdf_info(pdf_content)
        
        if "error" in info:
            raise HTTPException(status_code=400, detail=info["error"])
        
        return {
            "filename": file.filename,
            "info": info
        }
    
         except HTTPException:
         raise
     except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error reading PDF info: {str(e)}")

@app.post("/generate_citation_graph")
async def generate_citation_graph(
    topic: str = Query(..., description="Topic to search for articles"),
    graph_type: str = Query(default="author", description="Type of graph: 'author', 'keyword', or 'article'"),
    max_articles: int = Query(default=20, description="Maximum number of articles to include")
):
    """
    Generate citation graph visualization based on articles from a topic search
    """
    valid_types = ["author", "keyword", "article"]
    if graph_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"graph_type must be one of: {valid_types}")
    
    try:
        # Search for articles on the topic
        articles = crossref_tool._run(topic)
        
        # Limit the number of articles for performance
        articles = articles[:max_articles]
        
        if not articles:
            raise HTTPException(status_code=404, detail="No articles found for the given topic")
        
        # Create the appropriate network
        if graph_type == "author":
            network_data = citation_graph.create_author_network(articles)
        elif graph_type == "keyword":
            network_data = citation_graph.create_keyword_network(articles)
        elif graph_type == "article":
            network_data = citation_graph.create_article_similarity_network(articles)
        
        # Generate HTML visualization
        html_viz = citation_graph.generate_html_visualization(network_data, graph_type)
        
        return {
            "success": True,
            "topic": topic,
            "graph_type": graph_type,
            "total_articles": len(articles),
            "network_stats": network_data["stats"],
            "html_visualization": html_viz
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating citation graph: {str(e)}")

@app.post("/generate_graph_from_knowledge_base")
async def generate_graph_from_knowledge_base(
    graph_type: str = Query(default="author", description="Type of graph: 'author', 'keyword', or 'article'"),
    query: Optional[str] = Query(default=None, description="Optional query to filter articles"),
    max_results: int = Query(default=50, description="Maximum number of results to include")
):
    """
    Generate citation graph from existing knowledge base articles
    """
    valid_types = ["author", "keyword", "article"]
    if graph_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"graph_type must be one of: {valid_types}")
    
    try:
        # Get articles from knowledge base
        if query:
            # Search for specific articles
            docs = vector_store.search_similar(query=query, k=max_results)
            
            # Extract article data from document metadata
            articles = []
            seen_titles = set()
            
            for doc in docs:
                title = doc.metadata.get('title')
                if title and title not in seen_titles:
                    article = {
                        'title': title,
                        'authors': doc.metadata.get('authors', '').split(', ') if doc.metadata.get('authors') else [],
                        'year': doc.metadata.get('year'),
                        'journal': doc.metadata.get('journal', ''),
                        'risk_type': doc.metadata.get('risk_type', ''),
                        'level_of_analysis': doc.metadata.get('level_of_analysis', ''),
                        'abstract': doc.page_content[:500]  # Use chunk content as abstract
                    }
                    articles.append(article)
                    seen_titles.add(title)
        else:
            # This would require a method to get all articles from vector store
            # For now, return an error asking for a query
            raise HTTPException(status_code=400, detail="Please provide a query to filter articles from the knowledge base")
        
        if not articles:
            raise HTTPException(status_code=404, detail="No articles found in knowledge base")
        
        # Create the appropriate network
        if graph_type == "author":
            network_data = citation_graph.create_author_network(articles)
        elif graph_type == "keyword":
            network_data = citation_graph.create_keyword_network(articles)
        elif graph_type == "article":
            network_data = citation_graph.create_article_similarity_network(articles)
        
        # Generate HTML visualization
        html_viz = citation_graph.generate_html_visualization(network_data, graph_type)
        
        return {
            "success": True,
            "query": query,
            "graph_type": graph_type,
            "total_articles": len(articles),
            "network_stats": network_data["stats"],
            "html_visualization": html_viz
        }
    
         except HTTPException:
         raise
     except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error generating graph from knowledge base: {str(e)}")

@app.post("/run_monitoring")
async def run_article_monitoring(topics: Optional[List[str]] = None):
    """
    Run article monitoring manually for specified topics or default topics
    """
    try:
        result = await run_manual_monitoring(topics)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running article monitoring: {str(e)}")

@app.get("/monitoring_status")
async def get_monitoring_status():
    """
    Get current article monitoring status and configuration
    """
    try:
        status = article_monitor.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting monitoring status: {str(e)}")

@app.post("/monitoring_config")
async def update_monitoring_config(
    enabled: Optional[bool] = None,
    max_articles_per_topic: Optional[int] = None,
    max_days_since_publication: Optional[int] = None
):
    """
    Update article monitoring configuration
    """
    try:
        config_updates = {}
        if enabled is not None:
            config_updates["enabled"] = enabled
        if max_articles_per_topic is not None:
            config_updates["max_articles_per_topic"] = max_articles_per_topic
        if max_days_since_publication is not None:
            config_updates["max_days_since_publication"] = max_days_since_publication
        
        article_monitor.update_config(**config_updates)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "new_config": article_monitor.config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating monitoring config: {str(e)}")

@app.post("/monitoring_topics")
async def manage_monitoring_topics(
    action: str = Query(..., description="Action: 'add' or 'remove'"),
    topic: str = Query(..., description="Topic to add or remove")
):
    """
    Add or remove topics from monitoring
    """
    try:
        if action == "add":
            article_monitor.add_topic(topic)
            return {
                "success": True,
                "message": f"Topic '{topic}' added to monitoring",
                "topics": article_monitor.default_topics
            }
        elif action == "remove":
            success = article_monitor.remove_topic(topic)
            if success:
                return {
                    "success": True,
                    "message": f"Topic '{topic}' removed from monitoring",
                    "topics": article_monitor.default_topics
                }
            else:
                return {
                    "success": False,
                    "message": f"Topic '{topic}' not found in monitoring list",
                    "topics": article_monitor.default_topics
                }
        else:
            raise HTTPException(status_code=400, detail="Action must be 'add' or 'remove'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error managing monitoring topics: {str(e)}")