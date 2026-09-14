from app.agent.graph import agent
from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def run_query(query: str, user_id: int):

    try:
        # ====================================================
        # Create temporary user data
        # ====================================================

        temp_path = create_user_temp(user_id)

        print("=" * 70)
        print("TEMP USER DATA")
        print("=" * 70)
        print(f"User ID : {user_id}")
        print(f"File    : {temp_path}")
        print("=" * 70)

        # ====================================================
        # Run agent
        # ====================================================

        result = agent.invoke({
            "user_id": user_id,
            "query": query,
        })

        print("=" * 70)
        print(f"USER ID : {user_id}")
        print(f"QUERY   : {query}")
        print(f"INTENT  : {result.get('intent')}")
        print(f"REASON  : {result.get('intent_reason')}")
        print("-" * 70)
        print(result.get("response"))
        print("=" * 70)

    finally:
        # ====================================================
        # Delete temporary file
        # ====================================================

        delete_user_temp(user_id)

        print("=" * 70)
        print("TEMP USER DATA DELETED")
        print("=" * 70)
        print(f"User ID : {user_id}")
        print("=" * 70)


if __name__ == "__main__":

    print("=" * 70)
    print("CINE REASON ASSISTANT")
    print("=" * 70)

    # User ID
    while True:
        try:
            user_id = int(input("Enter User ID: "))
            break
        except ValueError:
            print("User ID must be an integer.")

    # Query
    query = input("Enter your question: ").strip()

    if not query:
        print("Question cannot be empty.")
        raise SystemExit(1)

    print()

    run_query(
        query=query,
        user_id=user_id,
    )