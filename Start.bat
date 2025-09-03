@echo off
setlocal enableextensions

REM ============================================
REM BidWERX Prompt Lab (MVP) - Windows Launcher
REM ============================================

REM 0) Go to the project folder
cd /d "C:\Users\trist\Documents\GitHub\PromptLab"

echo.
echo === BidWERX Prompt Lab Launcher ===
echo Working directory: %CD%
echo.

REM 1) Locate a working Python
set "PYEXE="

for /f "usebackq delims=" %%I in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do set "PYEXE=%%I"
if not defined PYEXE (
  for /f "usebackq delims=" %%I in (`py -c "import sys; print(sys.executable)" 2^>nul`) do set "PYEXE=%%I"
)
if not defined PYEXE (
  for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable)" 2^>nul`) do set "PYEXE=%%I"
)

if not defined PYEXE (
  echo [ERROR] No working Python found.
  pause
  exit /b 1
)

echo Using Python at: %PYEXE%
"%PYEXE%" -V
echo.

REM 2) Create venv if missing
if not exist ".venv" (
  echo [INFO] Creating virtual environment...
  "%PYEXE%" -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
) else (
  echo [INFO] Virtual environment already exists.
)

REM 3) Activate venv
if exist ".venv\Scripts\activate" (
  call ".venv\Scripts\activate"
) else (
  echo [ERROR] Could not find activation script: .venv\Scripts\activate
  pause
  exit /b 1
)

REM 4) Install/upgrade dependencies
if exist "requirements.txt" (
  echo [INFO] Installing dependencies...
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
  )
) else (
  echo [WARN] requirements.txt not found; skipping dependency install.
)

REM 5) Launch Streamlit app
echo [INFO] Launching BidWERX Prompt Lab...
streamlit run app.py

echo.
echo Press any key to close...
pause >nul
endlocal
