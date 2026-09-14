from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

RAW_DIR = (
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

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies_with_plots.csv"
TAGS_FILE = RAW_DIR / "tags.csv"

OUTPUT_FILE = (
    OUTPUT_DIR / "user_profiles.parquet"
)


# ============================================================
# CONFIG
# ============================================================

HIGH_RATING_THRESHOLD = 4.0
LOW_RATING_THRESHOLD = 3.0
TOP_GENRES = 2


# ============================================================
# HELPERS
# ============================================================

def split_genres(value):
    """
    Convert:

        Action|Comedy|Drama

    into:

        ['Action', 'Comedy', 'Drama']
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        genre.strip()
        for genre in value.split("|")
        if genre.strip()
    ]


def unique_preserve_order(values):
    """
    Remove duplicates while preserving order.
    """

    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


# ============================================================
# 1. LOAD RAW TABLES
# ============================================================

def load_raw_tables():
    print("=" * 70)
    print("STEP 1 - LOAD RAW TABLES")
    print("=" * 70)

    required_files = [
        RATINGS_FILE,
        MOVIES_FILE,
        TAGS_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    tags = pd.read_csv(TAGS_FILE)

    print(f"ratings.csv            : {len(ratings):,} rows")
    print(f"movies_with_plots.csv  : {len(movies):,} rows")
    print(f"tags.csv               : {len(tags):,} rows")

    return ratings, movies, tags


# ============================================================
# 2. PREPARE MOVIES TABLE
# ============================================================

def prepare_movies(movies):
    """
    Temporary movie table.

    Keeps only the information required for
    user-profile construction.
    """

    print()
    print("=" * 70)
    print("STEP 2 - PREPARE MOVIE TABLE")
    print("=" * 70)

    movies = movies.copy()

    movies["movieId"] = (
        movies["movieId"]
        .astype(int)
    )

    movies["genres_list"] = (
        movies["genres"]
        .apply(split_genres)
    )

    movies_temp = movies[
        [
            "movieId",
            "title",
            "genres",
            "genres_list",
        ]
    ].copy()

    print(
        f"Temporary movie table   : "
        f"{len(movies_temp):,} rows"
    )

    print(
        f"Unique movie IDs        : "
        f"{movies_temp['movieId'].nunique():,}"
    )

    return movies_temp


# ============================================================
# 3. PREPARE RATINGS TABLE
# ============================================================

def prepare_ratings(ratings):
    """
    Temporary ratings table.
    """

    print()
    print("=" * 70)
    print("STEP 3 - PREPARE RATINGS TABLE")
    print("=" * 70)

    ratings = ratings.copy()

    ratings["userId"] = (
        ratings["userId"]
        .astype(int)
    )

    ratings["movieId"] = (
        ratings["movieId"]
        .astype(int)
    )

    ratings["rating"] = (
        ratings["rating"]
        .astype(float)
    )

    print(
        f"Temporary ratings table : "
        f"{len(ratings):,} rows"
    )

    print(
        f"Unique users            : "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"Unique movies rated     : "
        f"{ratings['movieId'].nunique():,}"
    )

    return ratings


# ============================================================
# 4. JOIN RATINGS + MOVIES
# ============================================================

def create_user_movie_table(ratings, movies):
    """
    Join:

        ratings
            +
        movies

    Result:

        userId
        movieId
        rating
        title
        genres
        genres_list
    """

    print()
    print("=" * 70)
    print("STEP 4 - CREATE USER-MOVIE TABLE")
    print("=" * 70)

    user_movies = ratings.merge(
        movies,
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    missing_movies = (
        user_movies["title"]
        .isna()
        .sum()
    )

    print(
        f"Joined rows             : "
        f"{len(user_movies):,}"
    )

    print(
        f"Missing movie metadata  : "
        f"{missing_movies:,}"
    )

    if missing_movies > 0:
        raise ValueError(
            "Some ratings do not have "
            "matching movie metadata."
        )

    return user_movies


# ============================================================
# 5. CREATE HIGH-RATED TABLE
# ============================================================

def create_high_rated_table(user_movies):
    """
    Movies rated >= 4 stars.
    """

    print()
    print("=" * 70)
    print("STEP 5 - HIGH-RATED MOVIES")
    print("=" * 70)

    high_rated = user_movies[
        user_movies["rating"]
        >= HIGH_RATING_THRESHOLD
    ].copy()

    print(
        f"High-rated rows        : "
        f"{len(high_rated):,}"
    )

    print(
        f"Users with high ratings : "
        f"{high_rated['userId'].nunique():,}"
    )

    return high_rated


# ============================================================
# 6. CREATE LOW-RATED TABLE
# ============================================================

def create_low_rated_table(user_movies):
    """
    Movies rated <= 3 stars.
    """

    print()
    print("=" * 70)
    print("STEP 6 - LOW-RATED MOVIES")
    print("=" * 70)

    low_rated = user_movies[
        user_movies["rating"]
        <= LOW_RATING_THRESHOLD
    ].copy()

    print(
        f"Low-rated rows         : "
        f"{len(low_rated):,}"
    )

    return low_rated


# ============================================================
# 7. CREATE USER TAG TABLE
# ============================================================

def create_user_tags_table(tags):
    """
    Temporary table:

        userId -> unique tags
    """

    print()
    print("=" * 70)
    print("STEP 7 - USER TAGS")
    print("=" * 70)

    tags = tags.copy()

    tags["userId"] = (
        tags["userId"]
        .astype(int)
    )

    tags = tags[
        tags["tag"].notna()
    ].copy()

    tags["tag"] = (
        tags["tag"]
        .astype(str)
        .str.strip()
    )

    tags = tags[
        tags["tag"] != ""
    ]

    user_tags = (
        tags
        .groupby("userId", sort=True)["tag"]
        .apply(
            lambda values:
            unique_preserve_order(
                values.tolist()
            )
        )
        .to_dict()
    )

    print(
        f"Users with tags        : "
        f"{len(user_tags):,}"
    )

    print(
        f"Tag rows used           : "
        f"{len(tags):,}"
    )

    return user_tags


# ============================================================
# 8. HIGH-RATED GENRES
# ============================================================

def get_high_rated_genres(user_high_rated):
    """
    Get unique genres from movies
    rated >= 4 stars.
    """

    genres = []

    for movie_genres in (
        user_high_rated["genres_list"]
    ):
        genres.extend(movie_genres)

    return unique_preserve_order(
        genres
    )


# ============================================================
# 9. LOW-RATED GENRES
# ============================================================

def get_low_rated_genres(user_low_rated):
    """
    Get unique genres from movies
    rated <= 3 stars.
    """

    genres = []

    for movie_genres in (
        user_low_rated["genres_list"]
    ):
        genres.extend(movie_genres)

    return unique_preserve_order(
        genres
    )


# ============================================================
# 10. TOP 2 MOST WATCHED GENRES
# ============================================================

def get_top_2_genres(user_movies):
    """
    Count genres across all movies rated by the user.

    Ranking:

        1. Higher frequency
        2. Alphabetical order for ties
    """

    genre_counts = {}

    for movie_genres in (
        user_movies["genres_list"]
    ):
        for genre in movie_genres:
            genre_counts[genre] = (
                genre_counts.get(genre, 0)
                + 1
            )

    sorted_genres = sorted(
        genre_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return [
        genre
        for genre, _ in sorted_genres[
            :TOP_GENRES
        ]
    ]


# ============================================================
# 11. UNWATCHED MATCHING MOVIES
# ============================================================

def get_unwatched_matching_movie_ids(
    user_movies,
    movies,
    high_rated_genres,
):
    """
    Find movies that:

        1. The user has NOT rated
        2. The movie has at least one genre
           matching the user's high-rated genres

    Example:

        User high-rated genres:
            ['Action', 'Thriller']

        Candidate:
            Movie 100 -> Action
            Movie 200 -> Comedy
            Movie 300 -> Thriller

        Result:
            [100, 300]
    """

    if not high_rated_genres:
        return []

    # --------------------------------------------------------
    # Movies already rated by this user
    # --------------------------------------------------------

    watched_movie_ids = set(
        user_movies["movieId"]
        .astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # Movies not rated by this user
    # --------------------------------------------------------

    unwatched_movies = movies[
        ~movies["movieId"].isin(
            watched_movie_ids
        )
    ].copy()

    if unwatched_movies.empty:
        return []

    # --------------------------------------------------------
    # Match genre
    # --------------------------------------------------------

    high_genres = set(
        high_rated_genres
    )

    matching_movie_ids = []

    for _, movie in (
        unwatched_movies.iterrows()
    ):
        movie_genres = set(
            movie["genres_list"]
        )

        if movie_genres & high_genres:
            matching_movie_ids.append(
                int(movie["movieId"])
            )

    return unique_preserve_order(
        matching_movie_ids
    )


# ============================================================
# 12. BUILD FINAL USER PROFILES
# ============================================================

def build_user_profiles(
    user_movies,
    movies,
    user_tags,
):
    """
    Build final profile table.

    Final columns:

        user_id
        high_rated_genres
        high_rated_movies
        user_tags
        low_rated_genres
        unwatched_matching_movie_ids
        top_2_genres
    """

    print()
    print("=" * 70)
    print("STEP 8 - BUILD USER PROFILES")
    print("=" * 70)

    profiles = []

    for user_id, user_df in (
        user_movies
        .groupby("userId", sort=True)
    ):
        user_id = int(user_id)

        # ----------------------------------------------------
        # High-rated movies
        # ----------------------------------------------------

        high_rated = user_df[
            user_df["rating"]
            >= HIGH_RATING_THRESHOLD
        ].copy()

        high_rated_movies = (
            high_rated
            .sort_values(
                "rating",
                ascending=False,
            )
            ["title"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        # ----------------------------------------------------
        # High-rated genres
        # ----------------------------------------------------

        high_rated_genres = (
            get_high_rated_genres(
                high_rated
            )
        )

        # ----------------------------------------------------
        # Low-rated genres
        # ----------------------------------------------------

        low_rated = user_df[
            user_df["rating"]
            <= LOW_RATING_THRESHOLD
        ].copy()

        low_rated_genres = (
            get_low_rated_genres(
                low_rated
            )
        )

        # ----------------------------------------------------
        # User tags
        # ----------------------------------------------------

        tags = user_tags.get(
            user_id,
            [],
        )

        # ----------------------------------------------------
        # Top 2 genres
        # ----------------------------------------------------

        top_2_genres = (
            get_top_2_genres(
                user_df
            )
        )

        # ----------------------------------------------------
        # Unwatched matching movies
        # ----------------------------------------------------

        unwatched_matching_movie_ids = (
            get_unwatched_matching_movie_ids(
                user_movies=user_df,
                movies=movies,
                high_rated_genres=high_rated_genres,
            )
        )

        # ----------------------------------------------------
        # Final profile
        # ----------------------------------------------------

        profile = {
            "user_id": user_id,

            "high_rated_genres": list(
                high_rated_genres
            ),

            "high_rated_movies": list(
                high_rated_movies
            ),

            "user_tags": list(
                tags
            ),

            "low_rated_genres": list(
                low_rated_genres
            ),

            "unwatched_matching_movie_ids": list(
                unwatched_matching_movie_ids
            ),

            "top_2_genres": list(
                top_2_genres
            ),
        }

        profiles.append(profile)

    profiles_df = pd.DataFrame(
        profiles
    )

    return profiles_df


# ============================================================
# 13. VALIDATE FINAL SCHEMA
# ============================================================

def validate_schema(profiles):
    """
    Make sure the final Parquet contains exactly
    the required profile variables.
    """

    expected_columns = [
        "user_id",
        "high_rated_genres",
        "high_rated_movies",
        "user_tags",
        "low_rated_genres",
        "unwatched_matching_movie_ids",
        "top_2_genres",
    ]

    actual_columns = (
        profiles.columns.tolist()
    )

    if actual_columns != expected_columns:
        raise ValueError(
            "\nInvalid profile schema.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {actual_columns}"
        )

    print()
    print("=" * 70)
    print("STEP 9 - VALIDATE FINAL SCHEMA")
    print("=" * 70)

    for column in expected_columns:
        print(f"OK  {column}")


# ============================================================
# 14. SAVE PARQUET
# ============================================================

def save_profiles(profiles):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    print()
    print("=" * 70)
    print("STEP 10 - SAVE PARQUET")
    print("=" * 70)

    print(
        f"File    : {OUTPUT_FILE}"
    )

    print(
        f"Users   : {len(profiles):,}"
    )

    print(
        f"Columns : {len(profiles.columns)}"
    )


# ============================================================
# 15. VERIFY SAVED PARQUET
# ============================================================

def verify_saved_profiles():
    """
    Read the generated Parquet again and verify
    that the expected columns exist.
    """

    print()
    print("=" * 70)
    print("STEP 11 - VERIFY SAVED PARQUET")
    print("=" * 70)

    loaded = pd.read_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
    )

    expected_columns = [
        "user_id",
        "high_rated_genres",
        "high_rated_movies",
        "user_tags",
        "low_rated_genres",
        "unwatched_matching_movie_ids",
        "top_2_genres",
    ]

    assert (
        loaded.columns.tolist()
        == expected_columns
    )

    assert (
        loaded["user_id"]
        .is_unique
    )

    assert (
        loaded["user_id"]
        .notna()
        .all()
    )

    print(
        f"Loaded users    : "
        f"{len(loaded):,}"
    )

    print(
        f"Loaded columns  : "
        f"{len(loaded.columns)}"
    )

    print()

    for column in expected_columns:
        print(
            f"OK  {column:<40} "
            f"-> {type(loaded[column].iloc[0]).__name__}"
        )

    print()
    print(
        "Parquet verification: PASSED"
    )


# ============================================================
# 16. PRINT SAMPLE USERS
# ============================================================

def print_sample_profiles(
    profiles,
    n=5,
):
    """
    Print a few profiles for manual inspection.
    """

    print()
    print("=" * 70)
    print(
        f"SAMPLE USER PROFILES ({n})"
    )
    print("=" * 70)

    for _, row in (
        profiles.head(n).iterrows()
    ):
        print()
        print(
            f"User ID: {row['user_id']}"
        )

        print(
            "High-rated genres:"
        )
        print(
            f"  {row['high_rated_genres']}"
        )

        print(
            "High-rated movies:"
        )
        print(
            f"  {row['high_rated_movies'][:10]}"
        )

        print(
            "User tags:"
        )
        print(
            f"  {row['user_tags']}"
        )

        print(
            "Low-rated genres:"
        )
        print(
            f"  {row['low_rated_genres']}"
        )

        print(
            "Unwatched matching movie IDs:"
        )
        print(
            f"  {row['unwatched_matching_movie_ids'][:20]}"
        )

        print(
            "Top 2 genres:"
        )
        print(
            f"  {row['top_2_genres']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    ratings, movies, tags = (
        load_raw_tables()
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    movies = prepare_movies(
        movies
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    ratings = prepare_ratings(
        ratings
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    user_movies = (
        create_user_movie_table(
            ratings,
            movies,
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    high_rated = (
        create_high_rated_table(
            user_movies
        )
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    low_rated = (
        create_low_rated_table(
            user_movies
        )
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    user_tags = (
        create_user_tags_table(
            tags
        )
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    profiles = build_user_profiles(
        user_movies=user_movies,
        movies=movies,
        user_tags=user_tags,
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    validate_schema(
        profiles
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    save_profiles(
        profiles
    )

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    verify_saved_profiles()

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print_sample_profiles(
        profiles,
        n=5,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()