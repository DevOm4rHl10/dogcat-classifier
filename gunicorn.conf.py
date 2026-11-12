"""Gunicorn settings for serving the classifier."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# One worker keeps a single copy of the models in memory (~200 MB, under the 512 MB
# free-tier cap); threads let it handle a few uploads concurrently.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))

# Import app.py (and load the model) once, before workers start.
preload_app = True

timeout = 120   # free servers have a fraction of a CPU
keepalive = 5
accesslog = "-"
errorlog = "-"
