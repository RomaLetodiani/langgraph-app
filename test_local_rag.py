"""
Local test script for the agentic RAG system.
Run this to test your implementation without needing the LangGraph server.
"""

import asyncio

# Import your graph
from src.agent.graph import graph


async def test_agentic_rag_local():
    """Test the agentic RAG system locally."""

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        print("You can add it to a .env file in your project root:")
        print("OPENAI_API_KEY=your_api_key_here")
        return

    print("🚀 Testing Agentic RAG System Locally")
    print("=" * 50)

    # Test queries that demonstrate different behaviors
    test_cases = [
        {
            "name": "RAG Question - Should retrieve and use context",
            "query": "What does Lilian Weng say about types of reward hacking?",
            "expected_behavior": "Should retrieve docs and answer based on context",
        },
        {
            "name": "General Question - Should respond directly",
            "query": "Hello, how are you today?",
            "expected_behavior": "Should respond directly without retrieval",
        },
        {
            "name": "AI Topic - Might retrieve if relevant",
            "query": "Explain hallucination in AI systems",
            "expected_behavior": "Might retrieve if finds relevant context",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected_behavior']}")
        print("-" * 50)

        try:
            # Create input state
            input_state = {
                "messages": [{"role": "human", "content": test_case["query"]}]
            }

            # Run the graph
            print("Running graph...")

            # Stream the results
            step_count = 0
            async for step in graph.astream(input_state):
                step_count += 1
                for node_name, node_output in step.items():
                    print(f"  📍 Step {step_count} - Node: {node_name}")

                    # Print the messages in a readable format
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                print(
                                    f"     💬 {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}"
                                )
                            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                                print(f"     🔧 Tool call: {msg.tool_calls[0]['name']}")
                    print()

            print("✅ Test completed successfully!")

        except Exception as e:
            print(f"❌ Error during test: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()

        print("\n" + "=" * 50)

    print("🎉 All tests completed!")


async def simple_test():
    """Simple single test for quick verification."""
    print("🔬 Simple Agentic RAG Test")
    print("=" * 30)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    try:
        input_state = {
            "messages": [
                {
                    "role": "human",
                    "content": "What does Lilian Weng say about reward hacking?",
                }
            ]
        }

        print("🤖 Agent thinking...")
        result = await graph.ainvoke(input_state)

        print("\n💡 Final Response:")
        final_message = result["messages"][-1]
        print(final_message.content)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🌟 LangGraph Agentic RAG Local Tester")
    print("Choose your test:")
    print("1. Simple test (quick)")
    print("2. Comprehensive test (detailed)")

    choice = input("\nEnter your choice (1 or 2, default=1): ").strip() or "1"

    if choice == "2":
        asyncio.run(test_agentic_rag_local())
    else:
        asyncio.run(simple_test())
