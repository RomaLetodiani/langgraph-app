from langchain.chat_models import init_chat_model

from src.config.settings import settings, ConfigEnum


llm = init_chat_model(
    "azure_openai:gpt-o4-mini",
    azure_deployment=settings[ConfigEnum.AZURE_OPENAI_DEPLOYMENT_NAME],
    azure_endpoint=settings[ConfigEnum.AZURE_OPENAI_ENDPOINT],
    api_key=settings[ConfigEnum.AZURE_OPENAI_API_KEY],
    api_version=settings[ConfigEnum.AZURE_OPENAI_API_VERSION],
)
