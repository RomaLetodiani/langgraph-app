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


# Load environment values
raw_config: dict[ConfigEnum, str | None] = {
    key: os.getenv(key.value) for key in ConfigEnum
}

# Validate presence
missing = [k.value for k, v in raw_config.items() if not v]
if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}")

# Safe to assume non-None after this point
settings: dict[ConfigEnum, str] = {k: v for k, v in raw_config.items() if v is not None}
