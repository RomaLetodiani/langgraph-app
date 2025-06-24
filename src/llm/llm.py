from langchain.chat_models import init_chat_model
from src.config.config import config, ConfigEnum


llm = init_chat_model(
    "azure_openai:gpt-o4-mini",
    azure_deployment=config[ConfigEnum.AZURE_OPENAI_DEPLOYMENT_NAME],
    azure_endpoint=config[ConfigEnum.AZURE_OPENAI_ENDPOINT],
    api_key=config[ConfigEnum.AZURE_OPENAI_API_KEY],
    api_version=config[ConfigEnum.AZURE_OPENAI_API_VERSION],
)
