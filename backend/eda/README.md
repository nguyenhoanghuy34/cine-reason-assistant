## 📊 Dataset Structure

```mermaid
erDiagram

    MOVIES {
        int movieId PK
        string title
        int year
        string genres
        string plot
    }

    RATINGS {
        int userId FK
        int movieId FK
        float rating
        int timestamp
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
        float tmdbId
    }

    USERS {
        int userId PK
    }

    USERS ||--o{ RATINGS : rates
    USERS ||--o{ TAGS : creates

    MOVIES ||--o{ RATINGS : receives
    MOVIES ||--o{ TAGS : has
    MOVIES ||--|| LINKS : maps_to
