import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.tools.personal_tools import get_user_profile
from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def test_user_15_profile():
    user_id = 15

    try:
        temp_path = create_user_temp(user_id)

        profile = get_user_profile(user_id)

        print("\n")
        print("=" * 80)
        print(f"USER PROFILE - USER {user_id}")
        print("=" * 80)

        for key, value in profile.items():
            print(f"\n{key}:")
            print("-" * 80)
            print(value)

        print("\n" + "=" * 80)
        print("PROFILE SUMMARY")
        print("=" * 80)

        print(f"User ID                  : {profile['user_id']}")
        print(f"Top 2 genres             : {profile['top_2_genres']}")
        print(f"High-rated genres        : {profile['high_rated_genres']}")
        print(f"Low-rated genres         : {profile['low_rated_genres']}")
        print(f"High-rated movies        : {profile['high_rated_movies']}")
        print(f"User tags                : {profile['user_tags']}")

        movie_ids = profile["unwatched_matching_movie_ids"]

        print(
            f"Unwatched matching IDs   : "
            f"{len(movie_ids)} movies"
        )

        print("=" * 80)

        assert profile["user_id"] == user_id

    finally:
        delete_user_temp(user_id)