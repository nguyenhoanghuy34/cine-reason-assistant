```mermaid
erDiagram

    MOVIES {
        int movieId PK
        string title
        string genres
    }

    MOVIES_WITH_PLOTS {
        int movieId PK, FK
        string title
        int year
        string genres
        text plot
    }

    RATINGS {
        int userId FK
        int movieId FK
        float rating
        int timestamp
    }

    USERS {
        int userId PK
    }

    TAGS {
        int userId FK
        int movieId FK
        string tag
        int timestamp
    }

    LINKS {
        int movieId FK
        int imdbId
        int tmdbId
    }

    MOVIES ||--o| MOVIES_WITH_PLOTS : "movieId"
    MOVIES ||--o{ RATINGS : "movieId"
    USERS ||--o{ RATINGS : "userId"
    MOVIES ||--o{ TAGS : "movieId"
    USERS ||--o{ TAGS : "userId"
    MOVIES ||--o{ LINKS : "movieId"
```
