from typing import Optional, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from ..config import settings

class ArticleAnalysisSchema(BaseModel):
    objective: str
    methodology: str
    key_variables: str
    risk_type: str
    level_of_analysis: str
    main_findings: str
    implications: str
    limitations: str

class LangChainModel:
    def __init__(self):
        self.base_model = AzureChatOpenAI(
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_API_ENDPOINT,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            model_name=settings.AZURE_OPENAI_MODEL_NAME,
        )

    async def generate(self, input: str, prompt: str, schema: Optional[BaseModel] = None) -> Dict[str, Any]:
        """
        Generate a response with optional schema validation using JsonOutputParser
        """
        parser = JsonOutputParser(pydantic_object=schema)
        prompt_template = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{prompt}\n",
            input_variables=["query", "prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt_template | self.base_model | parser
        response = await chain.ainvoke({"query": input, "prompt": prompt})
        return response

    async def analyze_article(self, abstract: str) -> Dict[str, Any]:
        """
        Analyze an article abstract and extract structured information
        """
        prompt = """
        First, if the abstract is not in English, translate it to English.
        Then, conduct a comprehensive academic analysis of the research article abstract. Extract and synthesize key information using established research analysis frameworks.

        Systematically identify and elaborate on:
        - The main objective of the research (including research questions or hypotheses if present)
        - The methodological approach (including research design, data collection methods, analytical techniques)
        - Key variables studied (both dependent and independent variables, control variables if mentioned)
        - Type of risk discussed (categorize according to standard risk classification frameworks)
        - Level of analysis (e.g., firm-level, industry-level, country-level, multi-level analysis)
        - Main findings (including statistical significance and effect sizes if mentioned)
        - Theoretical and practical implications (contributions to literature and recommendations for practitioners)
        - Limitations of the study (methodological constraints, generalizability issues, potential biases)

        Important: If the original abstract was not in English, ensure all analysis is done on the English translation.

        Ensure your analysis maintains academic rigor and follows systematic review principles.
        Structure your response in a clear JSON format, with detailed explanations for each component in English.
        """
        
        return await self.generate(
            input=abstract,
            prompt=prompt,
            schema=ArticleAnalysisSchema
        ) 