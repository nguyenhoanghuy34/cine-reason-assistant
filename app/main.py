from app.agent.graph import agent

from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


NUMBER_OF_QUESTIONS = 5


def print_result(
    question_number: int,
    query: str,
    result: dict,
) -> None:

    print()
    print("=" * 70)
    print(f"QUESTION {question_number}")
    print("=" * 70)

    print(f"You: {query}")

    print()
    print("LLM ANSWER")
    print("-" * 70)

    print(
        result.get(
            "response",
            "No response.",
        )
    )

    print("-" * 70)

    print()
    print("SOURCE / EVIDENCE")
    print("-" * 70)

    evidence = result.get(
        "evidence",
        {},
    )

    if not evidence:

        print(
            "No evidence available."
        )

        return

    print(
        f"User ID     : "
        f"{evidence.get('user_id')}"
    )

    print(
        f"Top 2 Genres: "
        f"{evidence.get('top_2_genres', [])}"
    )

    candidate_movies = evidence.get(
        "candidate_movies",
        [],
    )

    print(
        f"Candidates  : "
        f"{len(candidate_movies)}"
    )

    if candidate_movies:

        print()
        print("Candidate Movies:")

        for index, movie in enumerate(
            candidate_movies,
            start=1,
        ):

            print(
                f"{index}. "
                f"{movie.get('title')}"
            )

            print(
                f"   Movie ID : "
                f"{movie.get('movie_id')}"
            )

            print(
                f"   Year     : "
                f"{movie.get('year')}"
            )

            print(
                f"   Genres   : "
                f"{movie.get('genres')}"
            )


def run_chat(
    user_id: int,
) -> None:

    temp_path = None

    try:

        # ========================================================
        # 1. CREATE TEMP USER PROFILE
        # ========================================================

        print("=" * 70)
        print("CREATING USER PROFILE")
        print("=" * 70)

        temp_path = create_user_temp(
            user_id
        )

        print(
            f"User ID : {user_id}"
        )

        print(
            f"Profile : {temp_path}"
        )

        print("=" * 70)

        # ========================================================
        # 2. COLLECT 5 QUESTIONS
        # ========================================================

        print()
        print("=" * 70)
        print("PERSONAL TEST")
        print("=" * 70)

        print(
            "Enter 5 questions."
        )

        print(
            "The LLM will reason only after "
            "all 5 questions are entered."
        )

        print("=" * 70)

        questions = []

        for question_number in range(
            1,
            NUMBER_OF_QUESTIONS + 1,
        ):

            print()
            print(
                f"QUESTION "
                f"{question_number}/"
                f"{NUMBER_OF_QUESTIONS}"
            )

            query = input(
                "You: "
            ).strip()

            while not query:

                print(
                    "Question cannot be empty."
                )

                query = input(
                    "You: "
                ).strip()

            questions.append(
                query
            )

        # ========================================================
        # 3. SHOW COLLECTED QUESTIONS
        # ========================================================

        print()
        print("=" * 70)
        print("ALL QUESTIONS COLLECTED")
        print("=" * 70)

        for index, query in enumerate(
            questions,
            start=1,
        ):

            print(
                f"{index}. {query}"
            )

        # ========================================================
        # 4. CALL LLM ONLY ONCE
        # ========================================================

        print()
        print("=" * 70)
        print("LLM REASONING")
        print("=" * 70)

        result = agent.invoke(
            {
                "user_id": user_id,
                "query": "\n".join(
                    [
                        f"Question {index}: {query}"
                        for index, query
                        in enumerate(
                            questions,
                            start=1,
                        )
                    ]
                ),
            }
        )

        # ========================================================
        # 5. PRINT RESULT
        # ========================================================

        print()
        print("=" * 70)
        print("RESULT")
        print("=" * 70)

        print(
            result.get(
                "response",
                "No response.",
            )
        )

        # ========================================================
        # 6. PRINT SOURCE
        # ========================================================

        print()
        print("=" * 70)
        print("SOURCE / EVIDENCE")
        print("=" * 70)

        evidence = result.get(
            "evidence",
            {},
        )

        if not evidence:

            print(
                "No evidence available."
            )

        else:

            print(
                f"User ID     : "
                f"{evidence.get('user_id')}"
            )

            print(
                f"Top 2 Genres: "
                f"{evidence.get('top_2_genres', [])}"
            )

            candidate_movies = (
                evidence.get(
                    "candidate_movies",
                    [],
                )
            )

            print(
                f"Candidates  : "
                f"{len(candidate_movies)}"
            )

            for index, movie in enumerate(
                candidate_movies,
                start=1,
            ):

                print(
                    f"{index}. "
                    f"{movie.get('title')} "
                    f"({movie.get('year')})"
                )

                print(
                    f"   ID     : "
                    f"{movie.get('movie_id')}"
                )

                print(
                    f"   Genres : "
                    f"{movie.get('genres')}"
                )

        print()
        print("=" * 70)
        print("5 QUESTIONS COMPLETED")
        print("=" * 70)

    finally:

        # ========================================================
        # 7. DELETE TEMP PROFILE
        # ========================================================

        if temp_path is not None:

            delete_user_temp(
                user_id
            )

            print()
            print("=" * 70)
            print("CHAT ENDED")
            print("=" * 70)

            print(
                f"User ID : {user_id}"
            )

            print(
                "Profile : DELETED"
            )

            print("=" * 70)


if __name__ == "__main__":

    print("=" * 70)
    print("CINE REASON ASSISTANT")
    print("=" * 70)

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

        run_chat(
            user_id
        )

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