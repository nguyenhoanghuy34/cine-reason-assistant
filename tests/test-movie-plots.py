"""
Test movie plot mapping in movies_with_plots.csv.

Run:
    python tests/test-movie-plots.py
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "app"
    / "data"
    / "ml-latest-small-filtered"
    / "movies_with_plots.csv"
)


# ============================================================
# CONFIG
# ============================================================

MOVIES_TO_TEST = [
    "Twelve Monkeys",
    "Heat",
    "Blade Runner",
    "Toy Story",
    "The Godfather",
]


# ============================================================
# HELPERS
# ============================================================

def print_separator(char="=", length=100):
    print(char * length)


def find_movie(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """Find movies whose titles contain the requested title."""

    return df[
        df["title"].str.contains(
            title,
            case=False,
            na=False,
            regex=False,
        )
    ]


def print_movie(row: pd.Series):
    """Pretty-print one movie and its plot."""

    print_separator()

    print(f"Movie ID : {row.get('movieId')}")
    print(f"Title    : {row.get('title')}")
    print(f"Year     : {row.get('year', 'N/A')}")
    print(f"Genres   : {row.get('genres')}")

    print("\nPLOT")
    print("-" * 100)

    plot = row.get("plot")

    if pd.isna(plot) or not str(plot).strip():
        print("[NO PLOT]")
    else:
        print(str(plot))

    print()


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print_separator()
    print("MOVIE PLOT MAPPING TEST")
    print_separator()

    # --------------------------------------------------------
    # 1. Check file
    # --------------------------------------------------------

    print(f"\nDataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        print("\n[ERROR] Dataset not found.")
        return

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(f"Fields  : {list(df.columns)}")

    required_columns = {"movieId", "title", "genres", "plot"}

    missing = required_columns - set(df.columns)

    if missing:
        print(f"\n[ERROR] Missing columns: {missing}")
        return

    # --------------------------------------------------------
    # 3. Test selected movies
    # --------------------------------------------------------

    for movie_title in MOVIES_TO_TEST:

        print("\n")
        print_separator("#")
        print(f"SEARCH: {movie_title}")
        print_separator("#")

        result = find_movie(df, movie_title)

        if result.empty:
            print(f"[NOT FOUND] {movie_title}")
            continue

        print(f"Matches: {len(result)}\n")

        for _, row in result.iterrows():
            print_movie(row)

    # --------------------------------------------------------
    # 4. Special Twelve Monkeys diagnostic
    # --------------------------------------------------------

    print("\n")
    print_separator()
    print("TWELVE MONKEYS DIAGNOSTIC")
    print_separator()

    result = find_movie(df, "Twelve Monkeys")

    if result.empty:
        print("[FAIL] Twelve Monkeys not found.")
        return

    row = result.iloc[0]
    plot = str(row["plot"]).lower()

    suspicious_terms = [
        "lenny weinrib",
        "amanda",
        "linda ash",
        "greek chorus",
    ]

    expected_terms = [
        "james cole",
        "virus",
        "time",
    ]

    suspicious_found = [
        term for term in suspicious_terms
        if term in plot
    ]

    expected_found = [
        term for term in expected_terms
        if term in plot
    ]

    print(f"Movie ID : {row['movieId']}")
    print(f"Title    : {row['title']}")

    print("\nSuspicious terms:")
    for term in suspicious_found:
        print(f"  [FOUND] {term}")

    print("\nExpected Twelve Monkeys terms:")
    for term in expected_found:
        print(f"  [FOUND] {term}")

    print()

    if suspicious_found:
        print("[FAIL] Plot appears to belong to another movie.")
        print("       Dataset/movie-plot mapping is likely incorrect.")

    elif expected_found:
        print("[PASS] Plot appears consistent with Twelve Monkeys.")
        print("       If chatbot still returns the wrong plot,")
        print("       inspect the movie-summary retrieval tool.")

    else:
        print("[WARNING] Could not confidently identify the plot.")
        print("          Inspect the full plot printed above.")

    print_separator()


if __name__ == "__main__":
    main()