"""
AI Cloud Operations Agent

Phase 1 command-line application.
"""

from src.agent.agent import CloudOperationsAgent


def main():

    print("=" * 60)
    print("AI CLOUD OPERATIONS AGENT")
    print("=" * 60)

    agent = CloudOperationsAgent()

    question = input(
        "\nEnter your question: "
    )

    answer = agent.run(question)

    print("\n" + "=" * 60)
    print("AGENT ANSWER")
    print("=" * 60)

    print(answer)


if __name__ == "__main__":
    main()
