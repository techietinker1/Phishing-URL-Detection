"""WSGI/production entrypoint.

Run with:  python wsgi.py
Or let a process manager (systemd, Supervisor, etc.) invoke it.

We intentionally only import the existing app defined in app.py. Do NOT
instantiate a new Flask() here; that would lose routes and model state.
"""

from waitress import serve
from app import app  # noqa: E402 - import after comment

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)