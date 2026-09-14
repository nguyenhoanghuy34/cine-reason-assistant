from app.agent.graph import agent

from app.agent.tools.user_summary import (
    create_user_summary,
    delete_user_summary,
)

from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def run_query(
    query: str,
    user_id: int,
) -> None:

    temp_created = False
    summary_created = False

    try:
        # ---------------------------------------------------------
        # STEP 1: Create temporary user profile
        # ---------------------------------------------------------

        temp_path = create_user_temp(user_id)
        temp_created = True

        print("=" * 70)
        print("TEMP USER DATA")
        print("=" * 70)

        print(f"User ID : {user_id}")
        print(f"File    : {temp_path}")

        print("=" * 70)

        # ---------------------------------------------------------
        # STEP 2: Summarize user with Gemini
        # ---------------------------------------------------------

        print()
        print("=" * 70)
        print("CREATING USER SUMMARY")
        print("=" * 70)

        summary_path = create_user_summary(user_id)
        summary_created = True

        print()
        print(f"Summary : {summary_path}")

        print("-" * 70)

        summary = summary_path.read_text(
            encoding="utf-8"
        )

        print(summary)

        print("=" * 70)

        # ---------------------------------------------------------
        # STEP 3: Ask user question
        # ---------------------------------------------------------

        query = input(
            "Enter your question: "
        ).strip()

        if not query:
            print("Question cannot be empty.")
            return

        print()

        # ---------------------------------------------------------
        # STEP 4: Run agent
        # ---------------------------------------------------------

        result = agent.invoke(
            {
                "user_id": user_id,
                "query": query,
            }
        )

        # ---------------------------------------------------------
        # STEP 5: Print result
        # ---------------------------------------------------------

        print("=" * 70)
        print("AGENT RESULT")
        print("=" * 70)

        print(f"USER ID : {user_id}")
        print(f"QUERY   : {query}")
        print(f"INTENT  : {result.get('intent')}")
        print(f"REASON  : {result.get('intent_reason')}")

        print("-" * 70)

        print(
            result.get(
                "response",
                "No response.",
            )
        )

        print("=" * 70)

    finally:

        # ---------------------------------------------------------
        # STEP 6: Cleanup only when files were actually created
        # ---------------------------------------------------------

        if summary_created:
            delete_user_summary(user_id)

        if temp_created:
            delete_user_temp(user_id)

        print()
        print("=" * 70)
        print("TEMP DATA CLEANUP")
        print("=" * 70)

        if summary_created:
            print("Summary TXT : DELETED")
        else:
            print("Summary TXT : NOT CREATED")

        if temp_created:
            print("Profile     : DELETED")
        else:
            print("Profile     : NOT CREATED")

        print("=" * 70)


if __name__ == "__main__":

    print("=" * 70)
    print("CINE REASON ASSISTANT")
    print("=" * 70)

    while True:

        try:
            user_id = int(
                input("Enter User ID: ")
            )

            break

        except ValueError:
            print(
                "User ID must be an integer."
            )

    print()

    run_query(
        query="",
        user_id=user_id,
    )