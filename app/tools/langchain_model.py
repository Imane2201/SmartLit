from typing import Optional, Dict, Any, Tuple
from openai import AsyncAzureOpenAI
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
        self.client = AsyncAzureOpenAI(
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_API_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
        )

    async def generate(self, input: str, prompt: str, schema: Optional[BaseModel] = None) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Generate a response with optional schema validation using JsonOutputParser
        Returns tuple of (response, token_usage)
        """
        parser = JsonOutputParser(pydantic_object=schema)
        format_instructions = parser.get_format_instructions()
        
        messages = [
            {"role": "system", "content": format_instructions},
            {"role": "user", "content": f"{input}\n{prompt}"}
        ]

        response = await self.client.chat.completions.create(
            messages=messages,
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0,  # Lower temperature for more consistent structured output
        )

        # Extract token usage
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Parse the response content
        content = response.choices[0].message.content
        parsed_response = parser.parse(content)
        
        return parsed_response, token_usage

    async def analyze_article(self, abstract: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Analyze an article abstract and extract structured information
        Returns tuple of (analysis_result, token_usage)
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