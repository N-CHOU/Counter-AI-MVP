@echo off
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python not found. Install Python 3.9+ from python.org and check "Add to PATH".
  pause
  exit /b 1
)

if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)

call venv\Scripts\activate.bat
pip install --quiet -r requirements.txt

echo.
echo ===================================================
echo   Counter AI MVP - starting server
echo   Open http://localhost:5000 in your browser
echo   Press Ctrl+C to stop
echo ===================================================
echo.

python server.py
pause
