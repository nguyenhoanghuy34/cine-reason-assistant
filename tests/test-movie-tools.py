import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.agent.tools.personal_tools import get_user_profile
from app.agent.tools.movie_tools import get_candidate_movies
from app.data_extract.create_user_temp import (
    create_user_temp,
    delete_user_temp,
)


def test_get_candidate_movies():
    user_id = 1

    try:
        create_user_temp(user_id)

        profile = get_user_profile(user_id)

        movie_ids = profile["unwatched_matching_movie_ids"]

        candidates = get_candidate_movies(
            movie_ids=movie_ids,
            limit=10,
        )

        print("\n" + "=" * 70)
        print("CANDIDATE MOVIES")
        print("=" * 70)

        for movie in candidates:
            print(f"Movie ID : {movie['movie_id']}")
            print(f"Title    : {movie['title']}")
            print(f"Year     : {movie['year']}")
            print(f"Genres   : {movie['genres']}")
            print(f"Plot     : {movie['plot'][:300]}...")
            print("-" * 70)

        assert isinstance(candidates, list)

        for movie in candidates:
            assert "movie_id" in movie
            assert "title" in movie
            assert "genres" in movie
            assert "plot" in movie

    finally:
        delete_user_temp(user_id)