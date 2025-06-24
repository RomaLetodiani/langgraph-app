"""LangGraph Agentic RAG Implementation.

An intelligent RAG system that decides when to retrieve information,
grades document relevance, and can rewrite queries for better results.
"""

from __future__ import annotations

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.llm.llm import llm
from src.config.settings import settings, ConfigEnum


def create_retriever_tool_for_rag():
    """Create and configure the retriever tool for RAG."""
    # URLs for Lilian Weng's blog posts (as in the tutorial)
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # Load and process documents
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Create vector store and retriever
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, 
        embedding=AzureOpenAIEmbeddings(api_key=settings[ConfigEnum.AZURE_OPENAI_API_KEY], api_version=settings[ConfigEnum.AZURE_OPENAI_API_VERSION], azure_endpoint=settings[ConfigEnum.AZURE_OPENAI_ENDPOINT], azure_deployment=settings[ConfigEnum.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME])
    )
    retriever = vectorstore.as_retriever()

    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng's blog posts on AI topics like reward hacking, hallucination, and diffusion models.",
    )

    return retriever_tool


def generate_query_or_respond(state: MessagesState):
    """Generate a query using retriever tool or respond directly to the user."""
    # Get retriever tool
    retriever_tool = create_retriever_tool_for_rag()

    # Call model with tool binding
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


def grade_documents(state: MessagesState):
    """Grade the relevance of retrieved documents and route accordingly."""
    messages = state["messages"]

    # Get the last tool message (retrieved documents)
    last_message = messages[-1]

    # Simple grading logic - check if retrieved content seems relevant
    # In a real system, you might use an LLM to grade relevance
    if hasattr(last_message, "content") and last_message.content:
        content = str(last_message.content).lower()
        question = str(messages[0].content).lower()

        # Extract key terms from question for relevance check
        question_terms = set(question.split())
        content_terms = set(content.split())

        # Calculate overlap
        overlap = len(question_terms.intersection(content_terms))

        # If we have reasonable overlap or content length, consider it relevant
        if overlap >= 2 or len(content) > 100:
            return "generate_answer"
        else:
            return "rewrite_question"

    return "rewrite_question"


def rewrite_question(state: MessagesState):
    """Rewrite the user's question for better retrieval."""
    messages = state["messages"]
    question = messages[0].content

    rewrite_prompt = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:\n"
        "------- \n"
        f"{question}\n"
        "------- \n"
        "Formulate an improved question:"
    )

    response = llm.invoke([{"role": "user", "content": rewrite_prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


def generate_answer(state: MessagesState):
    """Generate final answer using retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    generate_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        f"Question: {question} \n"
        f"Context: {context}"
    )

    response = llm.invoke([{"role": "user", "content": generate_prompt}])
    return {"messages": [response]}


# Create the graph
def create_graph():
    """Create and configure the agentic RAG graph."""

    # Get retriever tool for the ToolNode
    retriever_tool = create_retriever_tool_for_rag()

    # Create the graph
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Add edges
    workflow.add_edge(START, "generate_query_or_respond")

    # Conditional edge: decide whether to retrieve or respond directly
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # After retrieval, grade documents and route accordingly
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    # Connect the flow
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


# Export the graph
graph = create_graph()
