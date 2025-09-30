@echo off
echo ========================================
echo    Invoice AI Privacy - Installer
echo    Full automatic installation
echo ========================================
echo:

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running as Administrator
) else (
    echo [ERROR] Please run as Administrator for installation!
    echo Right-click install.bat and select "Run as administrator"
    pause
    exit /b 1
)

echo:
echo Step 1: Installing Python
echo -------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Downloading Python 3.11...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe' -OutFile '%TEMP%\python-installer.exe'"
    echo [INSTALL] Installing Python (this may take a few minutes)...
    "%TEMP%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1
    del "%TEMP%\python-installer.exe"
    echo [OK] Python installed
) else (
    echo [OK] Python already installed
)

echo:
echo Step 2: Installing Ollama
echo -------------------------
if exist "%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe" (
    echo [OK] Ollama already installed
) else (
    echo [INSTALL] Downloading Ollama...
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.ai/download/OllamaSetup.exe' -OutFile '%TEMP%\OllamaSetup.exe'"
    echo [INSTALL] Installing Ollama...
    "%TEMP%\OllamaSetup.exe" /S
    del "%TEMP%\OllamaSetup.exe"
    echo [OK] Ollama installed
)

echo:
echo Step 3: Installing Git (if needed)
echo ----------------------------------
git --version >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Downloading Git...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe' -OutFile '%TEMP%\git-installer.exe'"
    echo [INSTALL] Installing Git...
    "%TEMP%\git-installer.exe" /SILENT
    del "%TEMP%\git-installer.exe"
    echo [OK] Git installed
) else (
    echo [OK] Git already installed
)

echo:
echo Step 4: Installing ngrok (optional)
echo -----------------------------------
where ngrok >nul 2>&1
if errorlevel 1 (
    echo [INSTALL] Downloading ngrok...
    powershell -Command "Invoke-WebRequest -Uri 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip' -OutFile '%TEMP%\ngrok.zip'"
    powershell -Command "Expand-Archive -Path '%TEMP%\ngrok.zip' -DestinationPath 'C:\Windows\System32\' -Force"
    del "%TEMP%\ngrok.zip"
    echo [OK] ngrok installed
) else (
    echo [OK] ngrok already available
)

echo:
echo Step 5: Setting up Python virtual environment
echo ---------------------------------------------
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat

echo [SETUP] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo:
echo Step 6: Setting up Ollama service
echo ---------------------------------
echo [START] Starting Ollama service...
start "Ollama Service" /min "%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 10 /nobreak >nul

echo [MODEL] Downloading Qwen2.5 7B Instruct Q4 model...
echo This is a large file (~4GB) and may take 10-30 minutes depending on your internet speed.
echo Please be patient...
"%USERPROFILE%\AppData\Local\Programs\Ollama\ollama.exe" pull qwen2.5:7b-instruct-q4_K_M

echo:
echo ========================================
echo    Installation Complete!
echo ========================================
echo:
echo What was installed:
echo - Python 3.11 (if not already present)
echo - Ollama AI runtime
echo - Git (if not already present)
echo - ngrok tunneling tool
echo - Qwen2.5 7B Instruct Q4 model (~4GB)
echo - Python virtual environment with dependencies
echo:
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Edit .env and set your API_KEY
echo 3. Run launch.bat to start the service
echo:
echo IMPORTANT: Restart your computer to ensure all PATH changes take effect!
echo:
echo [DEBUG] Installation finished - press any key to close
pause
echo [DEBUG] Script completed successfully
echo [DEBUG] You can now close this window
pause