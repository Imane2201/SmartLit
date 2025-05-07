from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_ENDPOINT: str
    AZURE_OPENAI_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str 
    AZURE_OPENAI_MODEL_NAME: str 

    SPREADSHEET_ID: str

    class Config:
        env_file = ".env"

settings = Settings() 