import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.tools.personal_tools import get_user_profile
from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def test_get_user_profile():
    user_id = 1

    try:
        # Create temporary user data for testing
        temp_path = create_user_temp(user_id)

        print("\n" + "=" * 70)
        print("TEMP USER DATA")
        print("=" * 70)
        print(f"User ID : {user_id}")
        print(f"File    : {temp_path}")
        print("=" * 70)

        # Call personal tool
        profile = get_user_profile(user_id)

        print("\n" + "=" * 70)
        print("USER PROFILE")
        print("=" * 70)

        for key, value in profile.items():
            print(f"{key}:")
            print(value)
            print("-" * 70)

        # Basic validation
        assert profile["user_id"] == user_id
        assert "high_rated_genres" in profile
        assert "high_rated_movies" in profile
        assert "user_tags" in profile
        assert "low_rated_genres" in profile
        assert "unwatched_matching_movie_ids" in profile
        assert "top_2_genres" in profile

    finally:
        # Always delete temporary data
        delete_user_temp(user_id)

        print("=" * 70)
        print("TEMP USER DATA DELETED")
        print("=" * 70)
        print(f"User ID : {user_id}")
        print("=" * 70)