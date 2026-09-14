from app.agent.graph import agent

from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def run_chat(user_id: int) -> None:

    temp_created = False

    try:
        # ==========================================================
        # 1. CREATE TEMP USER PROFILE
        # ==========================================================

        temp_path = create_user_temp(user_id)
        temp_created = True

        print("=" * 70)
        print("USER PROFILE")
        print("=" * 70)

        print(f"User ID : {user_id}")
        print(f"Profile : {temp_path}")

        print("=" * 70)

        # ==========================================================
        # 2. START CHAT
        # ==========================================================

        print()
        print("=" * 70)
        print("CHAT STARTED")
        print("=" * 70)

        print(f"User ID: {user_id}")
        print("Ask anything about movies.")
        print("Press Ctrl+C to exit.")
        print("=" * 70)

        while True:

            print()

            query = input(
                "You: "
            ).strip()

            if not query:
                continue

            # ------------------------------------------------------
            # Send user question to agent
            # ------------------------------------------------------

            result = agent.invoke(
                {
                    "user_id": user_id,
                    "query": query,
                }
            )

            # ------------------------------------------------------
            # Show answer
            # ------------------------------------------------------

            print()
            print("Assistant:")
            print("-" * 70)

            print(
                result.get(
                    "response",
                    "No response.",
                )
            )

            print("-" * 70)

    finally:

        # ==========================================================
        # 3. DELETE TEMP USER PROFILE
        # ==========================================================

        if temp_created:
            delete_user_temp(user_id)

        print()
        print("=" * 70)
        print("CHAT ENDED")
        print("=" * 70)

        if temp_created:
            print("Profile : DELETED")
        else:
            print("Profile : NOT CREATED")

        print("=" * 70)


if __name__ == "__main__":

    print("=" * 70)
    print("CINE REASON ASSISTANT")
    print("=" * 70)

    # ==============================================================
    # GET USER ID
    # ==============================================================

    while True:

        try:

            user_id = int(
                input(
                    "Enter User ID: "
                ).strip()
            )

            break

        except ValueError:

            print(
                "User ID must be an integer."
            )

    print()

    try:

        run_chat(user_id)

    except KeyboardInterrupt:

        print()
        print("Exiting...")

    except Exception as exc:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(exc)

        print("=" * 70)