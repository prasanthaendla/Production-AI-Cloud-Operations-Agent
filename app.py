"""
AI Cloud Operations Agent

Phase 1 command-line application.

The application entry point uses LangGraph as the
orchestration layer while preserving the existing
CloudOperationsAgent components.
"""

from src.agent.langgraph_workflow import CloudOperationsLangGraph


def main():

    print("=" * 60)
    print("AI CLOUD OPERATIONS AGENT")
    print("=" * 60)

    graph = CloudOperationsLangGraph()

    question = input(
        "\nEnter your question: "
    )

    result = graph.run(question)

    print("\n" + "=" * 60)
    print("AGENT ANSWER")
    print("=" * 60)

    if result.get("status") == "completed":
        print(result.get("final_answer", "No final answer generated."))
    else:
        print(
            result.get(
                "error",
                "The investigation could not be completed.",
            )
        )


if __name__ == "__main__":
    main()