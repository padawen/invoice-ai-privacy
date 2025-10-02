@echo off
echo ========================================
echo    Invoice AI Privacy - Launcher
echo    Starting local AI service
echo ========================================
echo:

REM Set environment variables
set OLLAMA_HOST=localhost:11434
set OLLAMA_MODEL=gemma2:9b
set OLLAMA_TIMEOUT=300
set DEBUG=false
set PORT=5000
set HOST=0.0.0.0
set OCR_LANGUAGE=hun+eng
set MAX_FILE_SIZE=52428800
set LOG_LEVEL=INFO

REM Check vision mode from .env
findstr "USE_VISION_MODEL=true" .env >nul 2>&1
if not errorlevel 1 (
    set VISION_MODE=enabled
    set MODEL_NAME=LLaVA 7B Vision
) else (
    set VISION_MODE=disabled
    set MODEL_NAME=Gemma2 9B
)

echo Configuration:
echo    Processing Mode: Vision=%VISION_MODE%
echo    Text Model: Gemma2 9B
echo    Vision Model: LLaVA 7B ^(for image processing^)
echo    OCR: Tesseract ^(Hungarian + English^)
echo    API Key: [From .env file]
echo    Ports: 5000 ^(Flask^), 11434 ^(Ollama^)
echo:

REM Find Ollama executable
set OLLAMA_PATH=%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe
if exist "%OLLAMA_PATH%" (
    set OLLAMA_EXE=%OLLAMA_PATH%
    echo [OK] Found Ollama at: %OLLAMA_EXE%
) else (
    echo [ERROR] Ollama not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please run install.bat first.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

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

REM Check if models exist
if "%VISION_MODE%"=="enabled" (
    echo [MODEL] Verifying LLaVA 7B vision model...
    "%OLLAMA_EXE%" list | findstr "llava:7b" >nul
    if errorlevel 1 (
        echo [ERROR] LLaVA model not found. Run: ollama pull llava:7b
        pause
        exit /b 1
    )
    echo [OK] Vision model ready
) else (
    echo [MODEL] Verifying Gemma2 9B model...
    "%OLLAMA_EXE%" list | findstr "gemma2:9b" >nul
    if errorlevel 1 (
        echo [ERROR] Model not found. Please run install.bat first.
        pause
        exit /b 1
    )
    echo [OK] Text model ready
)

REM Check and start ngrok if needed (optional)
echo [NGROK] Checking if ngrok is available...
where ngrok >nul 2>&1
if not errorlevel 1 (
    tasklist | findstr "ngrok.exe" >nul 2>&1
    if errorlevel 1 (
        echo [NGROK] Starting ngrok tunnel for port 5000...
        start "Ngrok Tunnel" /min ngrok http 5000
        timeout /t 5 /nobreak >nul
        echo [NGROK] Tunnel started - check http://localhost:4040 for URL
    ) else (
        echo [OK] Ngrok is already running
    )
) else (
    echo [INFO] ngrok not available - service will only be accessible locally
)

REM Start Flask application
echo [START] Starting Flask API on port 5000...
echo [INFO] Privacy API will be available at: http://localhost:5000
echo [INFO] Health check: http://localhost:5000/health
if exist ngrok.exe (
    echo [INFO] Ngrok tunnel: check http://localhost:4040 for public URL
)
echo:
echo [READY] Invoice AI Privacy is ready!
echo [READY] Press Ctrl+C to stop the service
echo:

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