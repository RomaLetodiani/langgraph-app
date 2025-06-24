import os
from enum import Enum


class ConfigEnum(Enum):
    AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_DEPLOYMENT_NAME = "AZURE_OPENAI_DEPLOYMENT_NAME"
    AZURE_OPENAI_API_VERSION = "AZURE_OPENAI_API_VERSION"
    TAVILY_API_KEY = "TAVILY_API_KEY"
    LANGSMITH_PROJECT = "LANGSMITH_PROJECT"
    LANGSMITH_API_KEY = "LANGSMITH_API_KEY"


config: dict[ConfigEnum, str | None] = {
    ConfigEnum.AZURE_OPENAI_API_KEY: os.getenv(ConfigEnum.AZURE_OPENAI_API_KEY.value),
    ConfigEnum.AZURE_OPENAI_ENDPOINT: os.getenv(ConfigEnum.AZURE_OPENAI_ENDPOINT.value),
    ConfigEnum.AZURE_OPENAI_DEPLOYMENT_NAME: os.getenv(
        ConfigEnum.AZURE_OPENAI_DEPLOYMENT_NAME.value
    ),
    ConfigEnum.AZURE_OPENAI_API_VERSION: os.getenv(
        ConfigEnum.AZURE_OPENAI_API_VERSION.value
    ),
    ConfigEnum.TAVILY_API_KEY: os.getenv(ConfigEnum.TAVILY_API_KEY.value),
    ConfigEnum.LANGSMITH_PROJECT: os.getenv(ConfigEnum.LANGSMITH_PROJECT.value),
    ConfigEnum.LANGSMITH_API_KEY: os.getenv(ConfigEnum.LANGSMITH_API_KEY.value),
}

if not all(config.values()):
    raise ValueError("Missing Azure OpenAI or Tavily or LangSmith API key")
