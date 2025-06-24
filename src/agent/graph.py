"""LangGraph Agentic RAG Implementation.

An intelligent RAG system that decides when to retrieve information,
grades document relevance, and can rewrite queries for better results.
"""

from __future__ import annotations


from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.llm.llm import llm
from src.utils.docs import docs_vector_store


def create_retriever_tool_for_rag():
    """Create and configure the retriever tool for RAG."""
    print("Creating retriever tool for RAG")

    retriever = docs_vector_store.as_retriever()

    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng's blog posts on AI topics like reward hacking, hallucination, and diffusion models.",
    )

    print("Retriever tool created")

    return retriever_tool


def generate_query_or_respond(state: MessagesState):
    """Generate a query using retriever tool or respond directly to the user."""
    print("Generating query or responding to user")
    # Get retriever tool
    retriever_tool = create_retriever_tool_for_rag()

    # Call model with tool binding
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    print("Query or response generated")
    return {"messages": [response]}


def grade_documents(state: MessagesState):
    """Grade the relevance of retrieved documents and route accordingly."""
    print("Grading documents")
    messages = state["messages"]

    # Get the last tool message (retrieved documents)
    last_message = messages[-1]

    # Simple grading logic - check if retrieved content seems relevant
    # In a real system, you might use an LLM to grade relevance
    if hasattr(last_message, "content") and last_message.content:
        content = str(last_message.content).lower()
        users_messages = [message for message in messages if message.type == "human"]
        user_content = str(" ".join([message.content for message in users_messages])).lower()

        # Extract key terms from question for relevance check
        question_terms = set(user_content.split())
        content_terms = set(content.split())

        # Calculate overlap
        overlap = len(question_terms.intersection(content_terms))

        # If we have reasonable overlap, consider it relevant
        if overlap >= 2:
            print("Documents are relevant")
            return "generate_answer"
        else:
            print("Documents are not relevant")
            return "rewrite_question"

    print("Documents are not relevant")
    return "rewrite_question"


def rewrite_question(state: MessagesState):
    """Rewrite the user's question for better retrieval with human-in-the-loop."""
    print("Rewriting question with human input")
    messages = state["messages"]
    # Use the human-provided rewrite
    rewritten_question = messages[-1].content.replace("Human rewrite:", "").strip()
    print(f"Using human-provided rewrite: {rewritten_question}")
    return {"messages": [{"role": "user", "content": rewritten_question}]}


def generate_answer(state: MessagesState):
    """Generate final answer using retrieved context."""
    print("Generating answer")
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
    print("Answer generated")
    return {"messages": [response]}


# Create the graph
def create_graph():
    """Create and configure the agentic RAG graph with human-in-the-loop."""
    print("Creating graph")
    # Get retriever tool for the ToolNode
    retriever_tool = create_retriever_tool_for_rag()

    # Create the graph
    workflow = StateGraph(MessagesState)
    print("Graph created")

    print("Adding nodes")
    # Add nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    print("Nodes added")

    print("Adding edges")
    # Add edges
    workflow.add_edge(START, "generate_query_or_respond")
    print("Edges added")

    print("Adding conditional edges")
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
    print("Conditional edges added")

    print("Connecting flow")
    # Connect the flow
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    print("Flow connected")

    print("Compiling graph with human-in-the-loop interrupt")    
    # Compile with interrupt before rewrite_question to enable human-in-the-loop
    graph = workflow.compile(interrupt_before=["rewrite_question"])
    print("Graph compiled with human-in-the-loop")
    return graph


# Export the graph
graph = create_graph()
