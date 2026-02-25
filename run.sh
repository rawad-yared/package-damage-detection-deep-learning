#!/usr/bin/env bash
# Launch the Streamlit app using the project's virtual environment
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found. Create it first:"
    echo "  python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

cd "$DIR/src"
exec "$PYTHON" -m streamlit run app.py "$@"
