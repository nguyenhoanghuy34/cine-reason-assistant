# Movie Discovery Assistant - Technical Report

## Problem decomposition

The task is divided into validated MovieLens access, deterministic user/content/collaborative analysis, explainable hybrid ranking, and a grounded natural-language orchestration layer. This prevents the conversational layer from calculating or inventing recommendation facts.

## Architecture

`MovieLensDataLoader` validates and caches the CSV files. `MovieRepository` is the only data-access boundary for upper layers. User profiling, TF-IDF retrieval, collaborative filtering, and hybrid ranking are independent services. LangChain tools expose bounded JSON-compatible outputs, and a LangGraph state machine runs:

```text
START -> understand query -> build plan -> execute tools -> collect evidence -> generate answer -> END
```

The graph state contains user ID, query, intent, plan, tool results, evidence, and final answer.

## Design decisions

- CSVs are loaded once per resolved dataset path; agents and services do not read them directly.
- Genres are normalized to tuples, timestamps are UTC datetimes, and TMDB/IMDb IDs are nullable.
- Similar-user scores use centered cosine/Pearson-style correlation over shared ratings, require at least two overlaps, and are shrunk by overlap count.
- Tool responses are bounded to avoid passing unnecessary plots or rating histories to the response layer.
- The graph is bounded and deterministic. Its renderer only uses tool evidence and explicitly states missing data or low confidence.

## Recommendation and evidence

The hybrid score is `0.40 collaborative + 0.30 content + 0.20 genre preference + 0.10 quality`. Content uses TF-IDF/cosine over title, genres, plot, and tags. Quality combines mean rating and log-scaled rating count. Recommendations exclude already rated movies and support query, required-genre, and excluded-genre constraints. Each item returns component scores, supporting high-rated movies when available, and a similar-user predicted rating.

## Evidence-based reasoning

Tools provide verified movie metadata, search results, user profiles, similar users, opinions, and recommendations. The agent is instructed not to invent movie metadata, ratings, users, or similarities. A similar-user opinion reports neighbor count, rated-neighbor count, average rating, positive ratio, and rating distribution. A missing user produces a structured tool error and a transparent response rather than a personalized result.

## Evaluation

The evaluator makes a temporal split per user: ratings are sorted by timestamp, old ratings enter train, and up to five recent ratings (maximum 20%) become test. Users need at least 10 ratings. The training repository and engine receive only train ratings, preventing held-out ratings from entering collaborative, popularity, or profile signals.

Relevance is defined as a held-out explicit rating `>= 4.0`; this is an evaluation convention, not a supplied binary label. Metrics are Precision@5/@10, Recall@5/@10, and NDCG@5/@10.

On the executed deterministic sample of 20 eligible users, 17 had at least one relevant held-out item and all six metrics were `0.0000`. This is a real result from the current implementation, not a fixed expected output. Qualitative ranking can appear plausible, but it did not recover those users' most recent held-out likes in this sample.

## Failure cases and limitations

- Cold-start users have no personalized fallback.
- Sparse histories and few overlaps weaken collaborative evidence.
- TF-IDF is lexical and cannot reliably handle semantic paraphrases.
- Quality includes a popularity component, so heavily rated movies can dominate.
- Sparse plots, tags, or ratings reduce retrieval and evidence quality.
- Per-user temporal holdout is not a strict single global-time production simulation.
- The LangGraph planner is rule-based and deterministic; it is prepared for tool-based LLM orchestration but does not require an external model in the demo.

## Future improvements

- Add a clearly labeled non-personalized cold-start fallback.
- Tune weights and candidate generation against a larger temporal evaluation and simple baselines.
- Add confidence/minimum-support constraints to genre and collaborative signals.
- Introduce semantic embeddings only after establishing a measured baseline.
- Add a global-time evaluation protocol and diversity/novelty metrics.
