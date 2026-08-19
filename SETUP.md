# Project Setup & Quickstart

Requires Python 3.11 or newer.

## 1. Environment Setup

From your terminal / PowerShell in the project directory:

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Running Automated Tests

Run unit and integration test suite with coverage:

```bash
pytest --cov=src --cov-report=term-missing tests/ -v
```

Run code quality and style linter:

```bash
ruff check src tests app
```

## 3. End-to-End CLI Pipeline

Reproduce all data cleaning, feature engineering, model training, risk scoring, and report generation in a single command:

```bash
# Run pipeline with fallback templates (no API key required)
python -m src.pipeline --skip-llm

# Run pipeline with live Gemini LLM executive briefing generation
# (Ensure GEMINI_API_KEY is defined in .env)
python -m src.pipeline
```

## 4. Launching the Interactive Web Application

```bash
streamlit run app/streamlit_app.py
```

For Jupyter notebooks, select the `.venv` interpreter as your Jupyter kernel.
