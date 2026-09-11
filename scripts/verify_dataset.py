#!/usr/bin/env python3
"""Verify the dataset is properly set up."""

import sys
from pathlib import Path

import pandas as pd


def verify_dataset() -> bool:
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    # Resolve data directory relative to the repository root
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "backend" / "data" / "ml-latest-small-filtered"

    if not data_dir.exists():
        print(f"Dataset directory not found: {data_dir}")
        print("(Run this script from anywhere — it resolves paths from the repo root)")
        return False

    print(f"Found: {data_dir}\n")

    files = {
        "movies_with_plots.csv": "Movies with plot summaries",
        "ratings.csv": "User ratings",
        "tags.csv": "User tags",
        "links.csv": "External links",
        "movies.csv": "Basic movie info",
    }

    for filename, description in files.items():
        filepath = data_dir / filename

        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(
                f"  {filename:<30} "
                f"{size_mb:>6.2f} MB  "
                f"{description}"
            )
        else:
            print(f"  {filename:<30} MISSING")
            return False

    print("\n" + "-" * 60)

    # Load dataset
    movies = pd.read_csv(data_dir / "movies_with_plots.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")
    tags = pd.read_csv(data_dir / "tags.csv")

    print(f"Movies:  {len(movies):>6,}")
    print(f"Ratings: {len(ratings):>6,}")
    print(f"Tags:    {len(tags):>6,}")
    print(f"Users:   {ratings['userId'].nunique():>6,}")

    # Plot coverage
    plots_available = movies["plot"].notna().sum()
    plot_lengths = movies["plot"].dropna().str.len()

    print(
        f"\nPlot summaries: "
        f"{plots_available}/{len(movies)} movies"
    )

    if not plot_lengths.empty:
        print(
            f"Plot length: "
            f"min={plot_lengths.min():.0f}, "
            f"avg={plot_lengths.mean():.0f}, "
            f"max={plot_lengths.max():.0f} chars"
        )

    # Rating coverage
    rated_movies = ratings["movieId"].nunique()
    movies_with_few = (
        ratings.groupby("movieId").size() < 5
    ).sum()

    tagged_movies = tags["movieId"].nunique()

    print(
        f"\nMovies with ratings: "
        f"{rated_movies}/{len(movies)}"
    )

    print(f"Movies with <5 ratings: {movies_with_few}")

    print(
        f"Movies with tags: "
        f"{tagged_movies}/{len(movies)}"
    )

    print("\n" + "-" * 60)
    print("Sample:")

    sample = movies.iloc[0]

    print(f"  {sample['title']} ({int(sample['year'])})")
    print(f"  Genres: {sample['genres']}")

    plot = sample["plot"]

    if pd.notna(plot):
        print(f"  Plot: {plot[:120]}...")
    else:
        print("  Plot: N/A")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    sys.exit(0 if verify_dataset() else 1)