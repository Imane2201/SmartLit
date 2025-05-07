from langchain.tools import BaseTool
import requests
from typing import List, Dict, Any, Optional
from pydantic import Field

class CrossRefSearchTool(BaseTool):
    name: str = Field(default="crossref_search")
    description: str = Field(default="Search for academic articles using CrossRef API")

    def _run(self, query: str) -> List[Dict[str, Any]]:
        base_url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": 10,  # Limit results
            "select": "title,author,published-print,container-title,abstract"
        }
        
        response = requests.get(base_url, params=params)
        results = response.json()["message"]["items"]
        
        articles = []
        for item in results:
            article = {
                "title": item.get("title", [None])[0],
                "authors": [author.get("given", "") + " " + author.get("family", "") 
                          for author in item.get("author", [])],
                "year": item.get("published-print", {}).get("date-parts", [[None]])[0][0],
                "journal": item.get("container-title", [None])[0],
                "abstract": item.get("abstract", "")
            }
            articles.append(article)
            
        return articles

    async def _arun(self, query: str) -> List[Dict[str, Any]]:
        # Implement async version if needed
        raise NotImplementedError("Async not implemented") 