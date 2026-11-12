"""Gunicorn settings for serving the classifier."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# One worker keeps a single copy of the model in memory (free tiers often cap RAM
# around 512 MB); threads let it handle a few uploads concurrently.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))

# Import app.py (and load the model) once, before workers start.
preload_app = True

timeout = 60
keepalive = 5
accesslog = "-"
errorlog = "-"
