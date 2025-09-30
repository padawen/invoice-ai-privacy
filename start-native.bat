@echo off
echo Invoice AI Privacy - Native Setup
echo ====================================

REM Set environment variables
set OLLAMA_HOST=localhost:11434
set OLLAMA_MODEL=mistral:7b-instruct
set OLLAMA_TIMEOUT=300
set DEBUG=false
set PORT=5000
set HOST=0.0.0.0
set OCR_LANGUAGE=eng
set MAX_FILE_SIZE=52428800
REM API_KEY should be set in .env file for security
set LOG_LEVEL=INFO

echo Configuration:
echo    Model: Mistral 7B Instruct (multilingual support)
echo    OCR: Tesseract (English)
echo    API Key: [From .env file]
echo    Ports: 5000 (Flask), 11434 (Ollama)
echo.

REM Find Ollama executable
set OLLAMA_PATH=%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe
if exist "%OLLAMA_PATH%" (
    set OLLAMA_EXE=%OLLAMA_PATH%
    echo [OK] Found Ollama at: %OLLAMA_EXE%
) else (
    REM Try system PATH as fallback
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ollama not found. Please install from: https://ollama.ai/download
        pause
        exit /b 1
    ) else (
        set OLLAMA_EXE=ollama
        echo [OK] Found Ollama in system PATH
    )
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Install Python dependencies if needed
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

echo [SETUP] Installing Python dependencies...
pip install -r requirements.txt

REM Check if Ollama is already running
echo [OLLAMA] Checking if Ollama is already running...
tasklist | findstr "ollama.exe" >nul 2>&1
if errorlevel 1 (
    echo [START] Starting Ollama service...
    start "Ollama Service" /min "%OLLAMA_EXE%" serve
    echo [WAIT] Waiting for Ollama to start...
    timeout /t 10 /nobreak >nul
) else (
    echo [OK] Ollama is already running
)

REM Check if Ollama is responding
echo [CHECK] Verifying Ollama is responding...
call :wait_for_ollama
if errorlevel 1 (
    echo [ERROR] Ollama failed to start properly
    pause
    exit /b 1
)
echo [OK] Ollama is responding

REM Pull model if needed
echo [MODEL] Checking for Mistral 7B Instruct model...
"%OLLAMA_EXE%" list | findstr "mistral:7b-instruct" >nul
if errorlevel 1 (
    echo [MODEL] Pulling Mistral 7B Instruct model...
    "%OLLAMA_EXE%" pull mistral:7b-instruct
)

REM Check and start ngrok if needed
echo [NGROK] Checking if ngrok is already running...
tasklist | findstr "ngrok.exe" >nul 2>&1
if errorlevel 1 (
    echo [NGROK] Starting ngrok tunnel for port 5000...
    start "Ngrok Tunnel" /min ngrok http 5000
    timeout /t 5 /nobreak >nul
    echo [NGROK] Tunnel started - check http://localhost:4040 for URL
) else (
    echo [OK] Ngrok is already running
)

REM Start Flask application
echo [START] Starting Flask API on port 5000...
echo [INFO] Privacy API will be available at: http://localhost:5000
echo [INFO] Ngrok tunnel: check http://localhost:4040 for public URL
echo.
echo [READY] Invoice AI Privacy is ready!
echo.

python app.py
goto :eof

:wait_for_ollama
set /a attempts=0
:wait_loop
set /a attempts+=1
"%OLLAMA_EXE%" list >nul 2>&1
if not errorlevel 1 exit /b 0
if %attempts% geq 10 exit /b 1
echo [WAIT] Attempt %attempts%/10 - Ollama not ready yet...
timeout /t 2 /nobreak >nul
goto wait_loop