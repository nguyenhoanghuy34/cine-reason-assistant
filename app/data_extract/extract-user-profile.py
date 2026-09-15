from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Subject\HOME_TEST\cine-reason-assistant"
)

RAW_DIR = PROJECT_ROOT / "app" / "data" / "ml-latest-small-filtered"

OUTPUT_DIR = PROJECT_ROOT / "app" / "data" / "clean-data"

RATINGS_FILE = RAW_DIR / "ratings.csv"
MOVIES_FILE = RAW_DIR / "movies_with_plots.csv"
TAGS_FILE = RAW_DIR / "tags.csv"

OUTPUT_FILE = OUTPUT_DIR / "user_profiles.parquet"


# ============================================================
# CONFIG
# ============================================================

HIGH_RATING_THRESHOLD = 4.0
LOW_RATING_THRESHOLD = 3.0
TOP_GENRES = 2

# 1 <= watch_count <= 2 -> underexposed
UNDEREXPOSED_WATCH_THRESHOLD = 2


# ============================================================
# HELPERS
# ============================================================

def split_genres(value):
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
    print()
    print("=" * 70)
    print("STEP 2 - PREPARE MOVIE TABLE")
    print("=" * 70)

    movies = movies.copy()

    movies["movieId"] = movies["movieId"].astype(int)

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
        f"Unique movie IDs         : "
        f"{movies_temp['movieId'].nunique():,}"
    )

    return movies_temp


# ============================================================
# 3. PREPARE RATINGS TABLE
# ============================================================

def prepare_ratings(ratings):
    print()
    print("=" * 70)
    print("STEP 3 - PREPARE RATINGS TABLE")
    print("=" * 70)

    ratings = ratings.copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    print(
        f"Temporary ratings table : "
        f"{len(ratings):,} rows"
    )

    print(
        f"Unique users             : "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"Unique movies rated      : "
        f"{ratings['movieId'].nunique():,}"
    )

    return ratings


# ============================================================
# 4. JOIN RATINGS + MOVIES
# ============================================================

def create_user_movie_table(ratings, movies):
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
        f"Missing movie metadata   : "
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
    print()
    print("=" * 70)
    print("STEP 5 - HIGH-RATED MOVIES")
    print("=" * 70)

    high_rated = user_movies[
        user_movies["rating"] >= HIGH_RATING_THRESHOLD
    ].copy()

    print(
        f"High-rated rows         : "
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
    print()
    print("=" * 70)
    print("STEP 6 - LOW-RATED MOVIES")
    print("=" * 70)

    low_rated = user_movies[
        user_movies["rating"] <= LOW_RATING_THRESHOLD
    ].copy()

    print(
        f"Low-rated rows          : "
        f"{len(low_rated):,}"
    )

    return low_rated


# ============================================================
# 7. CREATE USER TAG TABLE
# ============================================================

def create_user_tags_table(tags):
    print()
    print("=" * 70)
    print("STEP 7 - USER TAGS")
    print("=" * 70)

    tags = tags.copy()

    tags["userId"] = tags["userId"].astype(int)

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
        f"Users with tags         : "
        f"{len(user_tags):,}"
    )

    print(
        f"Tag rows used            : "
        f"{len(tags):,}"
    )

    return user_tags


# ============================================================
# 8. HIGH-RATED GENRES
# ============================================================

def get_high_rated_genres(user_high_rated):
    genres = []

    for movie_genres in user_high_rated["genres_list"]:
        genres.extend(movie_genres)

    return unique_preserve_order(genres)


# ============================================================
# 9. LOW-RATED GENRES
# ============================================================

def get_low_rated_genres(user_low_rated):
    genres = []

    for movie_genres in user_low_rated["genres_list"]:
        genres.extend(movie_genres)

    return unique_preserve_order(genres)


# ============================================================
# 10. TOP 2 MOST WATCHED GENRES
# ============================================================

def get_top_2_genres(user_movies):
    genre_counts = {}

    for movie_genres in user_movies["genres_list"]:
        for genre in movie_genres:
            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
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
        for genre, _ in sorted_genres[:TOP_GENRES]
    ]


# ============================================================
# 11. GENRE WATCH COUNTS
# ============================================================

def get_genre_watch_counts(user_movies):
    genre_counts = {}

    for movie_genres in user_movies["genres_list"]:
        for genre in movie_genres:
            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
            )

    return dict(
        sorted(
            genre_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


# ============================================================
# 12. GENRE AVERAGE RATINGS
# ============================================================

def get_genre_avg_ratings(user_movies):
    genre_ratings = {}

    for _, row in user_movies.iterrows():
        rating = float(row["rating"])

        for genre in row["genres_list"]:
            if genre not in genre_ratings:
                genre_ratings[genre] = []

            genre_ratings[genre].append(rating)

    genre_avg_ratings = {}

    for genre, ratings in genre_ratings.items():
        genre_avg_ratings[genre] = round(
            sum(ratings) / len(ratings),
            3,
        )

    return dict(
        sorted(
            genre_avg_ratings.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


# ============================================================
# 13. GENRE HIGH-RATED COUNTS
# ============================================================

def get_genre_high_rated_counts(user_movies):
    genre_counts = {}

    for _, row in user_movies.iterrows():

        rating = float(row["rating"])

        if rating < HIGH_RATING_THRESHOLD:
            continue

        for genre in row["genres_list"]:
            genre_counts[genre] = (
                genre_counts.get(genre, 0) + 1
            )

    return dict(
        sorted(
            genre_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


# ============================================================
# 14. GENRE HIGH-RATING RATIO
# ============================================================

def get_genre_high_rating_ratio(
    genre_watch_counts,
    genre_high_rated_counts,
):
    ratios = {}

    for genre, watch_count in genre_watch_counts.items():

        high_count = (
            genre_high_rated_counts.get(
                genre,
                0,
            )
        )

        if watch_count == 0:
            ratio = 0.0
        else:
            ratio = high_count / watch_count

        ratios[genre] = round(
            ratio,
            3,
        )

    return dict(
        sorted(
            ratios.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


# ============================================================
# 15. ALL DATASET GENRES
# ============================================================

def get_all_genres(movies):
    all_genres = set()

    for movie_genres in movies["genres_list"]:
        all_genres.update(movie_genres)

    return sorted(all_genres)


# ============================================================
# 16. UNWATCHED GENRES
# ============================================================

def get_unwatched_genres(
    all_genres,
    genre_watch_counts,
):
    return [
        genre
        for genre in all_genres
        if genre_watch_counts.get(
            genre,
            0,
        ) == 0
    ]


# ============================================================
# 17. UNDEREXPOSED GENRES
# ============================================================

def get_underexposed_genres(
    all_genres,
    genre_watch_counts,
):
    return [
        genre
        for genre in all_genres
        if 1
        <= genre_watch_counts.get(
            genre,
            0,
        )
        <= UNDEREXPOSED_WATCH_THRESHOLD
    ]


# ============================================================
# 18. UNWATCHED MATCHING MOVIES
# ============================================================

def get_unwatched_matching_movie_ids(
    user_movies,
    movies,
    high_rated_genres,
):
    if not high_rated_genres:
        return []

    watched_movie_ids = set(
        user_movies["movieId"]
        .astype(int)
        .tolist()
    )

    unwatched_movies = movies[
        ~movies["movieId"].isin(
            watched_movie_ids
        )
    ].copy()

    if unwatched_movies.empty:
        return []

    high_genres = set(
        high_rated_genres
    )

    matching_movie_ids = []

    for _, movie in unwatched_movies.iterrows():

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
# 19. BUILD FINAL USER PROFILES
# ============================================================

def build_user_profiles(
    user_movies,
    movies,
    user_tags,
):
    print()
    print("=" * 70)
    print("STEP 8 - BUILD USER PROFILES")
    print("=" * 70)

    all_genres = get_all_genres(movies)

    print(
        f"All dataset genres      : "
        f"{len(all_genres):,}"
    )

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
            user_df["rating"] >= HIGH_RATING_THRESHOLD
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
        # Low-rated movies
        # ----------------------------------------------------

        low_rated = user_df[
            user_df["rating"] <= LOW_RATING_THRESHOLD
        ].copy()

        # ----------------------------------------------------
        # Low-rated genres
        # ----------------------------------------------------

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
        # Genre watch counts
        # ----------------------------------------------------

        genre_watch_counts = (
            get_genre_watch_counts(
                user_df
            )
        )

        # ----------------------------------------------------
        # Genre average ratings
        # ----------------------------------------------------

        genre_avg_ratings = (
            get_genre_avg_ratings(
                user_df
            )
        )

        # ----------------------------------------------------
        # Genre high-rated counts
        # ----------------------------------------------------

        genre_high_rated_counts = (
            get_genre_high_rated_counts(
                user_df
            )
        )

        # ----------------------------------------------------
        # Genre high-rating ratio
        # ----------------------------------------------------

        genre_high_rating_ratio = (
            get_genre_high_rating_ratio(
                genre_watch_counts,
                genre_high_rated_counts,
            )
        )

        # ----------------------------------------------------
        # Unwatched genres
        # ----------------------------------------------------

        unwatched_genres = (
            get_unwatched_genres(
                all_genres,
                genre_watch_counts,
            )
        )

        # ----------------------------------------------------
        # Underexposed genres
        # ----------------------------------------------------

        underexposed_genres = (
            get_underexposed_genres(
                all_genres,
                genre_watch_counts,
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
            # Existing variables
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

            # New blind-spot variables
            "genre_watch_counts": dict(
                genre_watch_counts
            ),

            "genre_avg_ratings": dict(
                genre_avg_ratings
            ),

            "genre_high_rated_counts": dict(
                genre_high_rated_counts
            ),

            "genre_high_rating_ratio": dict(
                genre_high_rating_ratio
            ),

            "unwatched_genres": list(
                unwatched_genres
            ),

            "underexposed_genres": list(
                underexposed_genres
            ),
        }

        profiles.append(profile)

    profiles_df = pd.DataFrame(
        profiles
    )

    return profiles_df


# ============================================================
# 20. VALIDATE FINAL SCHEMA
# ============================================================

def validate_schema(profiles):

    expected_columns = [
        "user_id",
        "high_rated_genres",
        "high_rated_movies",
        "user_tags",
        "low_rated_genres",
        "unwatched_matching_movie_ids",
        "top_2_genres",
        "genre_watch_counts",
        "genre_avg_ratings",
        "genre_high_rated_counts",
        "genre_high_rating_ratio",
        "unwatched_genres",
        "underexposed_genres",
    ]

    actual_columns = profiles.columns.tolist()

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
# 21. SAVE PARQUET
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
# 22. VERIFY SAVED PARQUET
# ============================================================

def verify_saved_profiles():

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
        "genre_watch_counts",
        "genre_avg_ratings",
        "genre_high_rated_counts",
        "genre_high_rating_ratio",
        "unwatched_genres",
        "underexposed_genres",
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
# 23. PRINT SAMPLE USERS
# ============================================================

def print_sample_profiles(
    profiles,
    n=5,
):
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

        print(
            "Genre watch counts:"
        )
        print(
            f"  {row['genre_watch_counts']}"
        )

        print(
            "Genre average ratings:"
        )
        print(
            f"  {row['genre_avg_ratings']}"
        )

        print(
            "Genre high-rated counts:"
        )
        print(
            f"  {row['genre_high_rated_counts']}"
        )

        print(
            "Genre high-rating ratio:"
        )
        print(
            f"  {row['genre_high_rating_ratio']}"
        )

        print(
            "Unwatched genres:"
        )
        print(
            f"  {row['unwatched_genres']}"
        )

        print(
            "Underexposed genres:"
        )
        print(
            f"  {row['underexposed_genres']}"
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