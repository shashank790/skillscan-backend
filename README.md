# SkillScan backend

Flask API for resume analysis using Google Gemini. Used by the mobile app to analyze uploaded PDFs/images. Uses **uv** for installs and running.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh` on Mac/Linux)
- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey)

## Setup (one-time)

```bash
cd resume/backend
uv sync
```

Create a `.env` file in this directory:

```
GEMINI_API_KEY=your_key_here
```

## Run the server

```bash
cd resume/backend
uv run python server.py
```

Or with the port explicitly:

```bash
cd resume/backend
PORT=5050 uv run python server.py
```

No need to activate a venv — `uv run` uses the project environment. Server runs at `http://0.0.0.0:5050` (default port 5050).

- **GET /** — health check (`"backend alive"`)
- **POST /analyze** — form field `file` (PDF or image); returns JSON with `overall`, `strengths`, `weaknesses`, `improvements`, `rewrite_example`

## Production

Use gunicorn (included in dependencies):

```bash
cd resume/backend
uv run gunicorn -w 1 -b 0.0.0.0:5050 server:app
```

Adjust `-w` (workers) as needed.

## Legacy (pip)

If you don’t use uv, `requirements.txt` is still present; use `pip install -r requirements.txt` in a venv and run with `python server.py` or `gunicorn` as above.
