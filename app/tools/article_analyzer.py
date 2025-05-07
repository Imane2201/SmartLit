from .langchain_model import LangChainModel

class ArticleAnalyzer:
    def __init__(self):
        self.model = LangChainModel()
    
    async def analyze(self, abstract: str) -> dict:
        result = await self.model.analyze_article(abstract)
        return result 