# GitHub Personality Profiler

GitHub Personality Profiler is a FastAPI app that analyzes a public GitHub profile, extracts NLP and behavioral signals from repositories and contributions, classifies the profile into an engineer archetype, and generates a shareable Open Graph card.

## Features

- Scrapes up to 30 public repositories, README files, topics, commit messages, issue comments, and pull requests with retry handling.
- Extracts documentation, technical depth, sentiment, vocabulary, and behavioral features.
- Uses a saved ML model when available, with a rule-based fallback when it is not.
- Renders a dark GitHub-inspired UI with charts, share links, and downloadable profile cards.
- Logs analysis requests locally to `analytics.csv`.

## Project Structure

- `main.py`: FastAPI app entrypoint and shared app state.
- `routers/`: HTML page routes and JSON/image API routes.
- `services/`: GitHub scraping, NLP, classification, card generation, and orchestration logic.
- `templates/`: Jinja templates for the landing page and result view.
- `train.py`: Optional model training script.

## Run Instructions

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm`
3. `cp .env.example .env  # add your GitHub token`
4. `python train.py  # optional, generates the ML model`
5. `uvicorn main:app --reload`

## Notes

- GitHub API access requires a personal access token in `.env`.
- The in-memory cache keeps profile analyses for one hour.
- Generated cards are saved to `/tmp/{username}_card.png`.
