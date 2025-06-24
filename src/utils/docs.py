
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings

from src.config.settings import settings, ConfigEnum

print("Loading docs")

# URLs for Lilian Weng's blog posts (as in the tutorial)
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

print(f"{len(urls)} urls to load")

# Load and process documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

print(f"{len(docs_list)} docs loaded")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

print(f"{len(doc_splits)} doc splits created")

# Create vector store and retriever
print("Creating vector store")
docs_vector_store = InMemoryVectorStore.from_documents(
    documents=doc_splits, 
    embedding=AzureOpenAIEmbeddings(api_key=settings[ConfigEnum.AZURE_OPENAI_API_KEY], api_version=settings[ConfigEnum.AZURE_OPENAI_API_VERSION], azure_endpoint=settings[ConfigEnum.AZURE_OPENAI_ENDPOINT], azure_deployment=settings[ConfigEnum.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME])
)

print("Docs vector store created")