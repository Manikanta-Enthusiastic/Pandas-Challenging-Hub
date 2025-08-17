"""
Polarized Books Analysis 📚

This script analyzes reading sessions for books to identify titles
that show strong polarization in reader ratings.

A book is considered "polarized" if:
    - It has at least 5 reading sessions
    - It received both high ratings (>= 4) and low ratings (<= 2)
    - It shows a spread between highest and lowest ratings

The script calculates:
    - Rating spread (highest - lowest rating)
    - Polarization score (ratio of extreme ratings to total sessions)
"""

import pandas as pd

# ---------------------------
# Sample data for demonstration
# ---------------------------

books = pd.DataFrame(
    [
        [1, "The Great Gatsby", "F. Scott", "Fiction", 180],
        [2, "To Kill a Mockingbird", "Harper Lee", "Fiction", 281],
        [3, "1984", "George Orwell", "Dystopian", 328],
        [4, "Pride and Prejudice", "Jane Austen", "Romance", 432],
        [5, "The Catcher in the Rye", "J.D. Salinger", "Fiction", 277],
    ],
    columns=["book_id", "title", "author", "genre", "pages"],
).astype(
    {
        "book_id": "int64",
        "title": "string",
        "author": "string",
        "genre": "string",
        "pages": "int64",
    }
)

reading_sessions = pd.DataFrame(
    [
        [1, 1, "Alice", 50, 5],
        [2, 1, "Bob", 60, 1],
        [3, 1, "Carol", 40, 4],
        [4, 1, "David", 30, 2],
        [5, 1, "Emma", 45, 5],
        [6, 2, "Frank", 80, 4],
        [7, 2, "Grace", 70, 4],
        [8, 2, "Henry", 90, 5],
        [9, 2, "Ivy", 60, 4],
        [10, 2, "Jack", 75, 4],
        [11, 3, "Kate", 100, 2],
        [12, 3, "Liam", 120, 1],
        [13, 3, "Mia", 80, 2],
        [14, 3, "Noah", 90, 1],
        [15, 3, "Olivia", 110, 4],
        [16, 3, "Paul", 95, 5],
        [17, 4, "Quinn", 150, 3],
        [18, 4, "Ruby", 140, 3],
        [19, 5, "Sam", 80, 1],
        [20, 5, "Tara", 70, 2],
    ],
    columns=["session_id", "book_id", "reader_name", "pages_read", "session_rating"],
).astype(
    {
        "session_id": "int64",
        "book_id": "int64",
        "reader_name": "string",
        "pages_read": "int64",
        "session_rating": "int64",
    }
)

# ---------------------------
# Functions
# ---------------------------

def read_books(reading_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate reading sessions to identify books
    with sufficient activity and extreme ratings.
    """
    # Aggregate sessions: count, highest rating, lowest rating
    result = reading_sessions.groupby("book_id", as_index=False).agg(
        total_sessions=("session_id", "count"),
        highest_rating=("session_rating", "max"),
        lowest_rating=("session_rating", "min"),
    )

    # Count extreme ratings (<=2 or >=4)
    extreme_counts = (
        reading_sessions.groupby("book_id")["session_rating"]
        .apply(lambda x: ((x <= 2) | (x >= 4)).sum())
        .reset_index(name="extreme_ratings")
    )

    # Merge results
    result = result.merge(extreme_counts, on="book_id")

    # Apply filtering rules
    return result[
        (result["total_sessions"] >= 5)
        & (result["highest_rating"] >= 4)
        & (result["lowest_rating"] <= 2)
    ]


def merged_df(df: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """
    Combine aggregated results with book metadata.
    Adds rating spread and polarization score.
    """
    df["rating_spread"] = df["highest_rating"] - df["lowest_rating"]
    df["polarization_score"] = round(df["extreme_ratings"] / df["total_sessions"], 2)

    return (
        df.merge(books, on="book_id", how="inner")[
            [
                "book_id",
                "title",
                "author",
                "genre",
                "pages",
                "rating_spread",
                "polarization_score",
            ]
        ]
        .sort_values(by=["polarization_score", "title"], ascending=[False, False])
        .reset_index(drop=True)
    )


# ---------------------------
# Main execution
# ---------------------------

def main():
    pd_1 = read_books(reading_sessions)
    pd_2 = merged_df(pd_1, books)
    print(pd_2)


if __name__ == "__main__":
    main()
