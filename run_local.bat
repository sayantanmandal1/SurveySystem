@echo off
REM Local development startup script for AI Surveillance System

echo 🚀 Starting AI Surveillance System - Local Development Mode
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if dashboard file exists
if not exist "src\dashboard\app.py" (
    echo ❌ Dashboard file not found!
    pause
    exit /b 1
)

echo 🌐 Dashboard starting at http://localhost:8501
echo 📝 Press Ctrl+C to stop the server
echo ------------------------------------------------------------

REM Start Streamlit with localhost configuration
python -m streamlit run src\dashboard\app.py --server.port=8501 --server.address=localhost --browser.gatherUsageStats=false --server.enableCORS=false --server.enableXsrfProtection=true

echo.
echo 🛑 Dashboard stopped
pause