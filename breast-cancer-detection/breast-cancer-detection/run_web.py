"""
Launch the Flask website without the debug reloader.

Run:
    python run_web.py
"""

from api.app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
