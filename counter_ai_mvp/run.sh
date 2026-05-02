#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
  echo "python3 not found. Install Python 3.9+ from python.org"
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
pip install --quiet -r requirements.txt

echo ""
echo "==================================================="
echo "  Counter AI MVP — starting server"
echo "  Open http://localhost:5000 in your browser"
echo "  Press Ctrl+C to stop"
echo "==================================================="
echo ""

python3 server.py
