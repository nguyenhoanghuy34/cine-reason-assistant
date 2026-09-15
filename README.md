# TrustedAI Movie Discovery Assistant

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-Tools-1C3C3C)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)](https://console.groq.com/)
[![Evaluation](https://img.shields.io/badge/Evaluation-43%2F43%20Passed-2E7D32)](#evaluation)

An evidence-grounded movie discovery assistant built on a filtered MovieLens dataset. The assistant uses deterministic dataset tools for ratings, genres, plots, title resolution, content similarity, similar-user reasoning, and recommendation evidence. The LLM synthesizes retrieved evidence into a concise answer; it is not treated as the source of truth.

![Functional evaluation](images/A.png)

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Groq API Key](#groq-api-key)
- [Run the Assistant](#run-the-assistant)
- [Run Tests](#run-tests)
- [Evaluation](#evaluation)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)

## Features

- Personalized recommendations from rating history and genre preference.
- Movie title resolution for exact titles, aliases, typos, MovieLens title-order variants, and unknown titles.
- TF-IDF content retrieval over titles, genres, and plot summaries.
- Similar-user reasoning with precomputed user similarity and rating aggregation.
- Multi-turn memory with LangGraph checkpointing.
- Grounded answers with explicit evidence instead of unsupported LLM claims.
- Offline evaluation for functional checks, grounding checks, edge cases, and recommendation-quality diagnostics.

![Title resolution](images/B.png)

## Project Structure

```text
cine-reason-assistant/
├── app/
│   ├── main.py                         # CLI entry point
│   ├── config.py                       # Loads .env and GROQ_API_KEY
│   ├── agent/
│   │   ├── graph.py                    # LangGraph workflow and memory
│   │   ├── router/                     # Intent classification
│   │   ├── nodes/                      # General, personal, and related-user nodes
│   │   ├── llm/                        # Groq LLM client and prompts
│   │   └── tools/                      # Dataset evidence tools
│   └── data/
│       ├── ml-latest-small-filtered/   # Filtered MovieLens CSV files
│       └── clean-data/                 # Precomputed profiles and similar users
├── scripts/
│   ├── trustedai_evaluation_suite.py   # Main offline evaluation suite
│   ├── evaluate_evidence_quality.py    # Quantitative evidence checks
│   └── verify_dataset.py               # Dataset verification helper
├── tests/                              # Pytest tests
├── images/                             # Report screenshots and figures
├── Reports/REPORT.md                   # Additional report notes
├── requirements.txt
└── README.md
```

## Dataset

The project expects the dataset to be available locally under:

```text
app/data/ml-latest-small-filtered/
```

Required files:

- `movies.csv`: movie IDs, titles, and genres.
- `movies_with_plots.csv`: movie metadata with plot summaries.
- `ratings.csv`: explicit user ratings.
- `tags.csv`: user-provided movie tags.
- `links.csv`: external movie identifiers.

Precomputed files:

- `app/data/clean-data/user_profiles.parquet`
- `app/data/clean-data/user_similarity.parquet`

The core join keys are `movieId` for movie data and `userId` for user data.

## Installation

### 1. Open the Project

```powershell
cd D:\Subject\HOME_TEST\cine-reason-assistant
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## Groq API Key

The interactive assistant uses Groq for LLM inference through `langchain-groq`. You must create a Groq API key before running the chat app.

1. Go to [https://console.groq.com/](https://console.groq.com/).
2. Sign in or create an account.
3. Open the API Keys section.
4. Create a new API key.
5. Create a `.env` file in the project root.

Example `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The key is loaded in `app/config.py`. If it is missing, the app will stop with:

```text
GROQ_API_KEY not found in .env
```

## Run the Assistant

Start the CLI assistant:

```powershell
python -m app.main
```

You will be asked for:

- `User ID`: a MovieLens user ID, for example `1`, `15`, or `30`.
- `Session name`: press Enter for `default`, or reuse a previous name to continue memory.

Example questions:

```text
What movies have I rated highly?
What genres do I like the most?
What should I watch tonight?
What do similar users think about Pulp Fiction?
Recommend something like Toy Story but not Animation.
Tell me about The Shawshnk Redemption.
```

Exit the chat with:

```text
/exit
```

![Personal rating history](images/Personal_rating.png)

## Run Tests

Run the pytest suite:

```powershell
pytest
```

Run specific test files:

```powershell
pytest tests\test_agent_memory.py
pytest tests\test_irrelevant_question_guard.py
```

## Evaluation

Run the main TrustedAI offline evaluation suite:

```powershell
python scripts\trustedai_evaluation_suite.py
```

This suite does not call the LLM. It checks deterministic evidence retrieval and recommendation behavior directly against local dataset files.

Expected summary:

```text
A. Functional tests: Passed 10/10
B. Title-resolution tests: Passed 9/9
C. Evidence/grounding tests: Passed 4/4
D. Similar-user reasoning tests: Passed 4/4
E. Multi-turn and multi-signal tests: Passed 7/7
F. Failure / edge-case tests: Passed 9/9
Overall evidence/functional checks: Passed 43/43
```

![Grounding evaluation](images/C.png)
![Similar-user evaluation](images/D.png)

### What the Evaluation Functions Check

- `evaluate_functional()`: rating history, genre preferences, personalized recommendations, user-specific outputs, and sparse-user handling.
- `evaluate_content_search()`: theme search, time-travel search, cross-genre theme matches, plot comparison, and seed-title similarity.
- `evaluate_title_resolution()`: exact titles, MovieLens title-order variants, typos, aliases, near-duplicate titles, genre lookup, plot lookup, and unknown movies.
- `evaluate_blind_spots_and_grounding()`: genre blind spots, high-rated versus frequently watched genres, required evidence keys, and no-fabrication answer constraints.
- `evaluate_similar_users()`: similar-user IDs, similar-user ratings for target movies, average rating, rating count, consensus recommendations, and insufficient evidence.
- `evaluate_multi_signal_and_memory()`: previous recommendation explanation, combined user-history and similar-user evidence, genre constraints, positive seed movies, and multi-turn constraints.
- `evaluate_edge_cases()`: invalid user IDs, missing movies, low-confidence fuzzy matches, watched-movie exclusion, sparse history, missing tags, ambiguous titles, and contradictory constraints.
- `evaluate_offline_metrics()`: a separate ranking diagnostic that hides recent positive ratings and checks whether recommendations recover them.

![Multi-signal evaluation](images/E.png)
![Failure cases](images/F.png)

## Example Workflows

### Personalized Recommendation

The assistant retrieves the current user's profile, rating history, preferred genres, already-watched movies, and candidate recommendations. Watched movies are filtered out before the final answer is generated.

![Movie title example](images/Movie_title.png)

### Similar-User Reasoning

For questions such as "What do similar users think about Pulp Fiction?", the system retrieves similar users, resolves the target title, gathers their ratings, and reports aggregate evidence such as rating count and average rating.

![Similar user example](images/Similar_user.png)

### Grounded Failure Handling

If a movie is missing from the dataset or a title match is low-confidence, the assistant should not fabricate plot, genre, or rating evidence. It returns a conservative answer based on available data.

![Offline metric diagnostic](images/G.png)

## Troubleshooting

### `GROQ_API_KEY not found`

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your key from [https://console.groq.com/](https://console.groq.com/).

### PowerShell Cannot Activate the Virtual Environment

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Missing Dataset Files

Check that these files exist:

```text
app/data/ml-latest-small-filtered/movies.csv
app/data/ml-latest-small-filtered/movies_with_plots.csv
app/data/ml-latest-small-filtered/ratings.csv
app/data/ml-latest-small-filtered/tags.csv
app/data/clean-data/user_profiles.parquet
app/data/clean-data/user_similarity.parquet
```

### Evaluation Works but Chat Fails

The evaluation suite is mostly deterministic and can run without LLM calls. The chat assistant requires `GROQ_API_KEY` because it uses Groq to synthesize final responses.

## Notes

The system is strongest at grounded evidence retrieval and explanation. The recommendation ranker is intentionally simple and explainable; it is not yet a high-performance trained ranking model. Future improvements should focus on a stronger hybrid recommender while preserving evidence-grounded answers.
