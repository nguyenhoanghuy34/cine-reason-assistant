import pandas as pd


USER_PROFILES_PATH = r"D:\Subject\HOME_TEST\cine-reason-assistant\app\data\clean-data\user_profiles.parquet"
USER_SIMILARITY_PATH = r"D:\Subject\HOME_TEST\cine-reason-assistant\app\data\clean-data\user_similarity.parquet"


def print_profile_table(row):
    print("\n" + "=" * 80)
    print(f"USER PROFILES - USER {row['user_id']}")
    print("=" * 80)

    data = {
        "Field": [
            "user_id",
            "high_rated_genres",
            "high_rated_movies",
            "user_tags",
            "low_rated_genres",
            "unwatched_matching_movie_ids",
            "top_2_genres",
        ],
        "Value": [
            row["user_id"],
            row["high_rated_genres"],
            row["high_rated_movies"],
            row["user_tags"],
            row["low_rated_genres"],
            row["unwatched_matching_movie_ids"],
            row["top_2_genres"],
        ],
    }

    table = pd.DataFrame(data)
    print(table.to_string(index=False))


def print_similarity_table(row):
    print("\n" + "=" * 80)
    print(f"USER SIMILARITY - USER {row['user_id']}")
    print("=" * 80)

    data = {
        "Field": [
            "user_id",
            "related_user_ids",
        ],
        "Value": [
            row["user_id"],
            row["related_user_ids"],
        ],
    }

    table = pd.DataFrame(data)
    print(table.to_string(index=False))


# ============================================================
# USER PROFILES
# ============================================================

user_profiles = pd.read_parquet(USER_PROFILES_PATH)

print("\n" + "#" * 80)
print("USER PROFILES - FIRST 5 ROWS")
print("#" * 80)

for _, row in user_profiles.head(5).iterrows():
    print_profile_table(row)


# ============================================================
# USER SIMILARITY
# ============================================================

user_similarity = pd.read_parquet(USER_SIMILARITY_PATH)

print("\n" + "#" * 80)
print("USER SIMILARITY - FIRST 5 ROWS")
print("#" * 80)

for _, row in user_similarity.head(5).iterrows():
    print_similarity_table(row)


print("\n" + "#" * 80)
print("TEST COMPLETED")
print("#" * 80)