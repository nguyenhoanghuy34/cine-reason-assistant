from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "clean-data"
)

RATINGS_PATH = DATA_DIR / "ratings.csv"
MOVIES_PATH = DATA_DIR / "movies_with_plots.csv"

OUTPUT_PATH = OUTPUT_DIR / "user_similarity.parquet"


# ============================================================
# CONFIG
# ============================================================

MIN_HIGH_RATING = 4.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(
        MOVIES_PATH,
        usecols=["movieId", "genres"],
    )

    print(f"Ratings shape : {ratings.shape}")
    print(f"Movies shape  : {movies.shape}")

    return ratings, movies


# ============================================================
# PREPARE HIGH-RATED DATA
# ============================================================

def prepare_high_rated_data(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("PREPARING HIGH-RATED DATA")
    print("=" * 70)

    # Only consider movies rated 4 or 5 stars
    high_ratings = ratings[
        ratings["rating"] >= MIN_HIGH_RATING
    ].copy()

    print(f"High-rated records : {len(high_ratings)}")

    # Attach genres
    high_ratings = high_ratings.merge(
        movies,
        on="movieId",
        how="left",
    )

    # Remove movies without genre information
    high_ratings = high_ratings[
        high_ratings["genres"].notna()
    ].copy()

    # Split genres
    high_ratings["genre"] = high_ratings["genres"].str.split("|")

    high_ratings = high_ratings.explode("genre")

    high_ratings["genre"] = high_ratings["genre"].str.strip()

    # Remove invalid genres
    high_ratings = high_ratings[
        high_ratings["genre"].notna()
        & (high_ratings["genre"] != "")
        & (high_ratings["genre"] != "(no genres listed)")
    ].copy()

    print(f"Genre-level records : {len(high_ratings)}")

    return high_ratings


# ============================================================
# BUILD USER GENRE DATA
# ============================================================

def build_user_genre_data(
    high_ratings: pd.DataFrame,
) -> dict[int, dict[str, dict[str, int]]]:

    """
    Build:

    user_genre_data[user_id][genre] = {
        "five_star": number of 5-star ratings,
        "four_star": number of 4-star ratings,
    }

    Example:

    {
        1: {
            "Action": {
                "five_star": 3,
                "four_star": 2
            },
            "Comedy": {
                "five_star": 1,
                "four_star": 4
            }
        }
    }
    """

    print()
    print("=" * 70)
    print("BUILDING USER GENRE DATA")
    print("=" * 70)

    user_genre_data: dict[
        int,
        dict[str, dict[str, int]]
    ] = {}

    for row in high_ratings.itertuples(index=False):

        user_id = int(row.userId)
        genre = str(row.genre)
        rating = float(row.rating)

        if user_id not in user_genre_data:
            user_genre_data[user_id] = {}

        if genre not in user_genre_data[user_id]:
            user_genre_data[user_id][genre] = {
                "five_star": 0,
                "four_star": 0,
            }

        if rating == 5.0:
            user_genre_data[user_id][genre]["five_star"] += 1

        elif rating == 4.0:
            user_genre_data[user_id][genre]["four_star"] += 1

    print(f"Users found : {len(user_genre_data)}")

    return user_genre_data


# ============================================================
# BUILD GENRE -> USERS INDEX
# ============================================================

def build_genre_users(
    user_genre_data: dict[int, dict[str, dict[str, int]]],
) -> dict[str, set[int]]:

    """
    Build an index:

    genre_users[genre] = users who have rated
    at least one movie >= 4 stars in that genre.
    """

    print()
    print("=" * 70)
    print("BUILDING GENRE -> USERS INDEX")
    print("=" * 70)

    genre_users: dict[str, set[int]] = {}

    for user_id, genres in user_genre_data.items():

        for genre in genres:

            if genre not in genre_users:
                genre_users[genre] = set()

            genre_users[genre].add(user_id)

    print(f"Genres found : {len(genre_users)}")

    return genre_users


# ============================================================
# CALCULATE RELATED USERS
# ============================================================

def calculate_related_users(
    user_genre_data: dict[int, dict[str, dict[str, int]]],
    genre_users: dict[str, set[int]],
) -> dict[int, list[int]]:

    """
    For each user:

    1. Find users sharing at least one genre where both users
       have rated movies >= 4 stars.

    2. Rank related users by:
       - Number of shared 5-star ratings: DESC
       - Number of shared 4-star ratings: DESC
       - User ID: ASC

    Important:
    This keeps the original similarity definition.

    The only change is the ordering of related_user_ids.
    """

    print()
    print("=" * 70)
    print("CALCULATING RELATED USERS")
    print("=" * 70)

    result: dict[int, list[int]] = {}

    for user_id, current_genres in user_genre_data.items():

        # ----------------------------------------------------
        # Find all users sharing at least one high-rated genre
        # ----------------------------------------------------

        related_users: set[int] = set()

        for genre in current_genres:

            users_in_same_genre = genre_users.get(
                genre,
                set(),
            )

            related_users.update(users_in_same_genre)

        # Remove current user
        related_users.discard(user_id)

        # ----------------------------------------------------
        # Calculate ranking score for each related user
        # ----------------------------------------------------

        ranked_users = []

        for related_user_id in related_users:

            related_genres = user_genre_data.get(
                related_user_id,
                {},
            )

            shared_five_star = 0
            shared_four_star = 0

            # ------------------------------------------------
            # Only count genres shared by both users
            # ------------------------------------------------

            shared_genres = (
                set(current_genres.keys())
                & set(related_genres.keys())
            )

            for genre in shared_genres:

                current_genre_data = current_genres[genre]
                related_genre_data = related_genres[genre]

                # ------------------------------------------------
                # Count common 5-star ratings first
                # ------------------------------------------------
                shared_five_star += min(
                    current_genre_data["five_star"],
                    related_genre_data["five_star"],
                )

                # ------------------------------------------------
                # Count common 4-star ratings
                # ------------------------------------------------
                shared_four_star += min(
                    current_genre_data["four_star"],
                    related_genre_data["four_star"],
                )

            ranked_users.append(
                (
                    related_user_id,
                    shared_five_star,
                    shared_four_star,
                )
            )

        # ----------------------------------------------------
        # SORT
        #
        # Priority:
        #   1. More shared 5-star ratings
        #   2. More shared 4-star ratings
        #   3. Smaller user ID
        # ----------------------------------------------------

        ranked_users.sort(
            key=lambda x: (
                -x[1],
                -x[2],
                x[0],
            )
        )

        result[user_id] = [
            user_id_data[0]
            for user_id_data in ranked_users
        ]

    print(f"Users processed : {len(result)}")

    return result


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

def create_output_dataframe(
    related_users: dict[int, list[int]],
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("CREATING OUTPUT")
    print("=" * 70)

    rows = []

    for user_id, related_user_ids in related_users.items():

        rows.append(
            {
                "user_id": user_id,
                "related_user_ids": related_user_ids,
            }
        )

    output = pd.DataFrame(rows)

    output = output.sort_values(
        "user_id"
    ).reset_index(drop=True)

    return output


# ============================================================
# VALIDATION
# ============================================================

def validate_output(
    output: pd.DataFrame,
    user_genre_data: dict[int, dict[str, dict[str, int]]],
) -> None:

    print()
    print("=" * 70)
    print("VALIDATING OUTPUT")
    print("=" * 70)

    assert "user_id" in output.columns
    assert "related_user_ids" in output.columns

    assert len(output) == len(user_genre_data)

    for row in output.itertuples(index=False):

        user_id = int(row.user_id)
        related_user_ids = row.related_user_ids

        current_genres = set(
            user_genre_data[user_id].keys()
        )

        for related_user_id in related_user_ids:

            assert related_user_id != user_id

            related_genres = set(
                user_genre_data[related_user_id].keys()
            )

            # Must share at least one high-rated genre
            assert (
                current_genres
                & related_genres
            )

    print("Validation passed.")


# ============================================================
# PRINT SAMPLE
# ============================================================

def print_sample(
    output: pd.DataFrame,
    user_genre_data: dict[int, dict[str, dict[str, int]]],
    num_users: int = 5,
    num_related_users: int = 10,
) -> None:

    print()
    print("=" * 70)
    print("SAMPLE RESULT")
    print("=" * 70)

    for row in output.head(num_users).itertuples(index=False):

        user_id = int(row.user_id)

        print()
        print(f"USER {user_id}")
        print("-" * 70)

        current_genres = user_genre_data[user_id]

        print(
            f"High-rated genres: "
            f"{list(current_genres.keys())}"
        )

        print()
        print(
            f"Top {num_related_users} related users:"
        )

        for rank, related_user_id in enumerate(
            row.related_user_ids[:num_related_users],
            start=1,
        ):

            related_genres = user_genre_data[
                related_user_id
            ]

            shared_genres = (
                set(current_genres.keys())
                & set(related_genres.keys())
            )

            shared_five_star = 0
            shared_four_star = 0

            for genre in shared_genres:

                shared_five_star += min(
                    current_genres[genre]["five_star"],
                    related_genres[genre]["five_star"],
                )

                shared_four_star += min(
                    current_genres[genre]["four_star"],
                    related_genres[genre]["four_star"],
                )

            print(
                f"{rank:2d}. "
                f"User {related_user_id:<4d} | "
                f"5★ common: {shared_five_star:<3d} | "
                f"4★ common: {shared_four_star:<3d}"
            )


# ============================================================
# SAVE
# ============================================================

def save_output(
    output: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("SAVING OUTPUT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows    : {len(output)}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("USER SIMILARITY EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    ratings, movies = load_data()

    # --------------------------------------------------------
    # 2. Keep ratings >= 4 and attach genres
    # --------------------------------------------------------

    high_ratings = prepare_high_rated_data(
        ratings,
        movies,
    )

    # --------------------------------------------------------
    # 3. Build user -> genre -> rating counts
    # --------------------------------------------------------

    user_genre_data = build_user_genre_data(
        high_ratings
    )

    # --------------------------------------------------------
    # 4. Build genre -> users index
    # --------------------------------------------------------

    genre_users = build_genre_users(
        user_genre_data
    )

    # --------------------------------------------------------
    # 5. Find and rank related users
    # --------------------------------------------------------

    related_users = calculate_related_users(
        user_genre_data,
        genre_users,
    )

    # --------------------------------------------------------
    # 6. Create output
    # --------------------------------------------------------

    output = create_output_dataframe(
        related_users
    )

    # --------------------------------------------------------
    # 7. Validate
    # --------------------------------------------------------

    validate_output(
        output,
        user_genre_data,
    )

    # --------------------------------------------------------
    # 8. Print sample
    # --------------------------------------------------------

    print_sample(
        output,
        user_genre_data,
        num_users=5,
        num_related_users=10,
    )

    # --------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------

    save_output(output)

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()