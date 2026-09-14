from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Subject\HOME_TEST\cine-reason-assistant")

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


# ============================================================
# HELPERS
# ============================================================

def split_genres(value):
    """
    Convert:
        'Action|Comedy|Drama'

    Into:
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

    Always returns a Python list.
    """

    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def ensure_python_list(value):
    """
    Ensure a value is always a native Python list.

    This prevents numpy.ndarray from leaking into
    the user profile data.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if hasattr(value, "tolist"):
        converted = value.tolist()

        if isinstance(converted, list):
            return converted

    return list(value)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("Loading raw data")
    print("=" * 70)

    if not RATINGS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {RATINGS_FILE}"
        )

    if not MOVIES_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {MOVIES_FILE}"
        )

    if not TAGS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {TAGS_FILE}"
        )

    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
    tags = pd.read_csv(TAGS_FILE)

    print(f"Ratings : {len(ratings):,}")
    print(f"Movies  : {len(movies):,}")
    print(f"Tags    : {len(tags):,}")

    return ratings, movies, tags


# ============================================================
# PREPARE MOVIE DATA
# ============================================================

def prepare_movies(movies):
    movies = movies.copy()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    movies = movies[
        [
            "movieId",
            "title",
            "genres_list",
        ]
    ]

    return movies


# ============================================================
# PREPARE TAG DATA
# ============================================================

def prepare_tags(tags):
    """
    Build:

        userId -> list of unique tags

    Example:

        18 -> [
            'quirky',
            'dark comedy',
            'thought-provoking'
        ]
    """

    tags = tags.copy()

    tags = tags[
        tags["tag"].notna()
    ]

    tags["tag"] = (
        tags["tag"]
        .astype(str)
        .str.strip()
    )

    tags = tags[
        tags["tag"] != ""
    ]

    tags_by_user = (
        tags
        .groupby("userId")["tag"]
        .apply(
            lambda values: unique_preserve_order(
                values.tolist()
            )
        )
        .to_dict()
    )

    # Explicitly guarantee native Python lists.
    tags_by_user = {
        int(user_id): ensure_python_list(tags_list)
        for user_id, tags_list in tags_by_user.items()
    }

    return tags_by_user


# ============================================================
# BUILD USER PROFILES
# ============================================================

def build_user_profiles(ratings, movies, tags):

    # --------------------------------------------------------
    # Join ratings with movie metadata
    # --------------------------------------------------------

    user_movies = ratings.merge(
        movies,
        on="movieId",
        how="left",
        validate="many_to_one",
    )

    missing_movies = user_movies["title"].isna().sum()

    if missing_movies > 0:
        print(
            f"Warning: {missing_movies:,} rating rows "
            f"have no matching movie metadata."
        )

    # --------------------------------------------------------
    # Prepare tag lookup
    # --------------------------------------------------------

    tags_by_user = prepare_tags(tags)

    # --------------------------------------------------------
    # Build profiles
    # --------------------------------------------------------

    profiles = []

    for user_id, user_df in user_movies.groupby(
        "userId",
        sort=True,
    ):

        user_id = int(user_id)

        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        rating_count = int(len(user_df))

        avg_rating = round(
            float(user_df["rating"].mean()),
            3,
        )

        # ====================================================
        # WATCHED MOVIES
        # ====================================================

        watched_movie_ids = (
            user_df["movieId"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )

        watched_movie_ids = ensure_python_list(
            watched_movie_ids
        )

        # ====================================================
        # HIGH-RATED MOVIES >= 4
        # ====================================================

        high_rated = user_df[
            user_df["rating"] >= HIGH_RATING_THRESHOLD
        ]

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

        high_rated_movies = ensure_python_list(
            high_rated_movies
        )

        # ====================================================
        # HIGH-RATED GENRES
        # ====================================================

        high_genres = []

        for genres in high_rated["genres_list"]:

            genres = ensure_python_list(genres)

            high_genres.extend(genres)

        high_rated_genres = unique_preserve_order(
            high_genres
        )

        high_rated_genres = ensure_python_list(
            high_rated_genres
        )

        # ====================================================
        # LOW-RATED GENRES <= 3
        # ====================================================

        low_rated = user_df[
            user_df["rating"] <= LOW_RATING_THRESHOLD
        ]

        low_genres = []

        for genres in low_rated["genres_list"]:

            genres = ensure_python_list(genres)

            low_genres.extend(genres)

        low_rated_genres = unique_preserve_order(
            low_genres
        )

        low_rated_genres = ensure_python_list(
            low_rated_genres
        )

        # ====================================================
        # TOP 2 MOST WATCHED GENRES
        # ====================================================

        genre_counts = {}

        for genres in user_df["genres_list"]:

            genres = ensure_python_list(genres)

            for genre in genres:

                genre_counts[genre] = (
                    genre_counts.get(genre, 0) + 1
                )

        top_2_genres = [
            genre
            for genre, _ in sorted(
                genre_counts.items(),
                key=lambda x: (-x[1], x[0]),
            )[:TOP_GENRES]
        ]

        top_2_genres = ensure_python_list(
            top_2_genres
        )

        # ====================================================
        # USER TAGS
        # ====================================================

        user_tags = tags_by_user.get(
            user_id,
            [],
        )

        user_tags = ensure_python_list(
            user_tags
        )

        # ====================================================
        # PROFILE ROW
        # ====================================================

        profile = {
            "user_id": user_id,

            "rating_count": int(
                rating_count
            ),

            "avg_rating": float(
                avg_rating
            ),

            "high_rated_movies": list(
                high_rated_movies
            ),

            "high_rated_genres": list(
                high_rated_genres
            ),

            "low_rated_genres": list(
                low_rated_genres
            ),

            "user_tags": list(
                user_tags
            ),

            "top_2_genres": list(
                top_2_genres
            ),

            "watched_movie_ids": list(
                watched_movie_ids
            ),
        }

        profiles.append(profile)

    return pd.DataFrame(profiles)


# ============================================================
# VALIDATE PROFILE TYPES
# ============================================================

def validate_profile_types(profiles):
    """
    Validate that every list-based profile field contains
    a native Python list.
    """

    list_columns = [
        "high_rated_movies",
        "high_rated_genres",
        "low_rated_genres",
        "user_tags",
        "top_2_genres",
        "watched_movie_ids",
    ]

    print()
    print("=" * 70)
    print("Validating profile types")
    print("=" * 70)

    for column in list_columns:

        invalid_rows = []

        for index, value in profiles[column].items():

            if not isinstance(value, list):
                invalid_rows.append(index)

        if invalid_rows:

            raise TypeError(
                f"Column '{column}' contains "
                f"non-list values at rows: "
                f"{invalid_rows[:10]}"
            )

        print(
            f"OK  {column:<25} -> list"
        )


# ============================================================
# SAVE
# ============================================================

def save_profiles(profiles):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_profile_types(
        profiles
    )

    profiles.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    print()
    print("=" * 70)
    print("Saved user profiles")
    print("=" * 70)

    print(
        f"File   : {OUTPUT_FILE}"
    )

    print(
        f"Users  : {len(profiles):,}"
    )

    print(
        f"Columns: {len(profiles.columns)}"
    )

    print()
    print("Columns:")

    for column in profiles.columns:
        print(f"  - {column}")


# ============================================================
# VERIFY SAVED PARQUET
# ============================================================

def verify_saved_profiles():
    print()
    print("=" * 70)
    print("Verifying saved Parquet")
    print("=" * 70)

    loaded = pd.read_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
    )

    list_columns = [
        "high_rated_movies",
        "high_rated_genres",
        "low_rated_genres",
        "user_tags",
        "top_2_genres",
        "watched_movie_ids",
    ]

    def to_python_list(value):
        """
        Convert numpy.ndarray / tuple / other iterable
        into a native Python list.
        """
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if hasattr(value, "tolist"):
            value = value.tolist()

            if isinstance(value, list):
                return value

        return list(value)

    # --------------------------------------------------------
    # Normalize list columns after reading Parquet
    # --------------------------------------------------------

    for column in list_columns:
        loaded[column] = loaded[column].apply(
            to_python_list
        )

    # --------------------------------------------------------
    # Check types
    # --------------------------------------------------------

    for column in list_columns:

        invalid_rows = []

        for index, value in loaded[column].items():

            if not isinstance(value, list):
                invalid_rows.append(index)

        if invalid_rows:
            raise TypeError(
                f"After reading Parquet, column "
                f"'{column}' contains non-list values "
                f"at rows: {invalid_rows[:10]}"
            )

        sample_value = loaded[column].iloc[0]

        print(
            f"OK  {column:<25} "
            f"-> {type(sample_value).__name__}"
        )

    print()
    print("Parquet verification: PASSED")


# ============================================================
# MAIN
# ============================================================

def main():

    ratings, movies, tags = load_data()

    movies = prepare_movies(
        movies
    )

    profiles = build_user_profiles(
        ratings=ratings,
        movies=movies,
        tags=tags,
    )

    save_profiles(
        profiles
    )

    verify_saved_profiles()

    print()
    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()