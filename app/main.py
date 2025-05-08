from fastapi import FastAPI
from typing import List
import json
from dotenv import load_dotenv

from app.tools.crossref import CrossRefSearchTool
from app.tools.article_analyzer import ArticleAnalyzer
from app.tools.sheets_handler import GoogleSheetsHandler

# Load environment variables
load_dotenv()

app = FastAPI()
crossref_tool = CrossRefSearchTool()
article_analyzer = ArticleAnalyzer()
sheets_handler = GoogleSheetsHandler()

@app.on_event("startup")
async def startup_event():
    # Initialize the Google Sheet with headers
    sheets_handler.initialize_sheet()

# Add a root endpoint for health check
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

@app.post("/search_articles")
async def search_articles(topic: str):
    # Search for articles
    articles = crossref_tool._run(topic)
    
    results = []
    for article in articles:
        if article["abstract"]:
            # Analyze article - note the await here
            analysis = await article_analyzer.analyze(article["abstract"])
            
            # Combine metadata with analysis
            full_article = {**article, **analysis}
            results.append(full_article)
    
    # Store results in Google Sheet
    if results:
        sheets_handler.append_articles(results)
    
    return results 