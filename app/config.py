from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_ENDPOINT: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str 
    AZURE_OPENAI_MODEL_NAME: str 
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    AZURE_OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"

    SPREADSHEET_ID: str
    
    # Vector store settings
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "smartlit_articles"

    class Config:
        env_file = ".env"

settings = Settings() 