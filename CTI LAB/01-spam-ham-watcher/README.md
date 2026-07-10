# 01 · Spam/Ham IMAP Watcher

Watches a Gmail inbox via IMAP, classifies each new email as spam or ham with a TF-IDF + Naive Bayes model, moves spam to `[Gmail]/Spam`, and serves a live dashboard.

## Quick Start

```bash
# Run the full pipeline (train model)
python run.py all

# Or run individual steps
python run.py train      # Download data and train the spam classifier
python run.py classify   # Classify a text message
python run.py watch      # Start Gmail IMAP watcher
python run.py web        # Launch FastAPI dashboard (port 8000)
```

## Files

| File | Purpose |
| --- | --- |
| `gmail_watcher.py` | IMAP poller. Classifies each `UNSEEN` message, marks ham as `\Seen`, copies spam to `[Gmail]/Spam`. Writes every classification to SQLite. |
| `email_parser.py` | Extracts the body (text/plain preferred, HTML stripped) from raw RFC822 bytes. |
| `spam_utils.py` | Loads the trained model, cleans text, returns `(is_spam, probability)`. |
| `train_text_model.py` | Downloads the UCI SMS Spam Collection, trains a TF-IDF + MultinomialNB pipeline, saves `spam_text_model.joblib`. |
| `classify.py` | One-shot CLI: `python classify.py "some text"` → prints SPAM/HAM + confidence. |
| `db.py` | SQLite layer (WAL mode) for the dashboard. |
| `app.py` | FastAPI server. `GET /` returns the HTML, `GET /api/data` returns JSON for the dashboard. |
| `static/index.html` | Dashboard UI. |
| `tokens.css` | Hallmark-locked design tokens for the dashboard. |
| `data/SMSSpamCollection` | UCI SMS Spam Collection (used to train the model). |
| `data/classifications.db` | Live store of watcher decisions (created on first run). |

## Setup

```bash
cd 01-spam-ham-watcher
python -m venv .venv
.venv\Scripts\activate
pip install joblib python-dotenv pandas scikit-learn fastapi 'uvicorn[standard]'
```

## Run

```bash
# 1. (One-time) train the model — only needed if spam_text_model.joblib is missing
python train_text_model.py

# 2. Watcher
python gmail_watcher.py

# 3. Dashboard (separate terminal)
uvicorn app:app --port 8000
# → open http://localhost:8000/
```

## Configuration

Copy `.env.template` to `.env` and fill in:

```
EMAIL_ADDRESS=your.email@gmail.com
EMAIL_PASSWORD=your-16-character-app-password
IMAP_SERVER=imap.gmail.com
IMAP_PORT=993
POLL_INTERVAL=30
BATCH_SIZE=50
MODEL_PATH=spam_text_model.joblib
LOG_LEVEL=INFO
DB_PATH=data/classifications.db
```

Gmail requires a 16-character **App Password** (Google Account → Security → 2-Step Verification → App passwords).
