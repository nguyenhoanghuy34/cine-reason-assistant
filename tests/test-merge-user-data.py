from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

CLEAN_DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
)

USER_PROFILES_PATH = (
    CLEAN_DATA_DIR
    / "user_profiles.parquet"
)

USER_SIMILARITY_PATH = (
    CLEAN_DATA_DIR
    / "user_similarity.parquet"
)


def main():
    print("=" * 100)
    print("USER DATA MERGE TEST")
    print("=" * 100)

    # ============================================================
    # 1. INPUT USER ID
    # ============================================================

    while True:
        try:
            user_id = int(
                input("\nEnter User ID: ").strip()
            )
            break
        except ValueError:
            print("User ID must be an integer.")

    # ============================================================
    # 2. LOAD TWO PARQUET FILES
    # ============================================================

    user_profiles = pd.read_parquet(
        USER_PROFILES_PATH
    )

    user_similarity = pd.read_parquet(
        USER_SIMILARITY_PATH
    )

    # ============================================================
    # 3. GET CURRENT USER FROM BOTH TABLES
    # ============================================================

    profile = user_profiles[
        user_profiles["user_id"] == user_id
    ].copy()

    similarity = user_similarity[
        user_similarity["user_id"] == user_id
    ].copy()

    if profile.empty:
        print(
            f"\nUser {user_id} not found in "
            "user_profiles.parquet."
        )
        return

    if similarity.empty:
        print(
            f"\nUser {user_id} not found in "
            "user_similarity.parquet."
        )
        return

    # ============================================================
    # 4. MERGE INTO TEMPORARY TABLE
    # ============================================================

    temp_table = profile.merge(
        similarity,
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    # ============================================================
    # 5. PRINT TEMPORARY TABLE
    # ============================================================

    print()
    print("=" * 100)
    print(f"TEMPORARY TABLE - USER {user_id}")
    print("=" * 100)

    print()
    print(
        f"Rows    : {temp_table.shape[0]}"
    )

    print(
        f"Columns : {temp_table.shape[1]}"
    )

    print()
    print(
        temp_table.to_string(
            index=False
        )
    )

    # ============================================================
    # 6. PRINT EACH FIELD MORE READABLY
    # ============================================================

    print()
    print("=" * 100)
    print(f"USER {user_id} - FIELD VALUES")
    print("=" * 100)

    row = temp_table.iloc[0]

    for column in temp_table.columns:
        print()
        print(f"[{column}]")
        print("-" * 100)
        print(row[column])

    # ============================================================
    # 7. SUMMARY
    # ============================================================

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(
        f"Temporary table for User {user_id} "
        "created in memory."
    )
    print("No new file was created.")
    print("=" * 100)


if __name__ == "__main__":
    main()