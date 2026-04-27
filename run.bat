@echo off
REM PQD launcher. Stays open. ASCII only. Verbose so you can see every step.

title PQD Launcher
echo ============================================================
echo  PQD Project Launcher
echo ============================================================
echo.

REM Move to the folder this script lives in
cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM ---- pre-flight checks ----
if not exist "venv312\Scripts\python.exe" (
    echo [ERROR] venv312\Scripts\python.exe not found.
    echo         Expected at: %CD%\venv312\Scripts\python.exe
    echo.
    pause
    exit /b 1
)
echo [OK] venv312 found.

if not exist "backend\app.py" (
    echo [ERROR] backend\app.py not found.
    echo.
    pause
    exit /b 1
)
echo [OK] backend\app.py found.

if not exist "dashboard\index.html" (
    echo [ERROR] dashboard\index.html not found.
    echo.
    pause
    exit /b 1
)
echo [OK] dashboard\index.html found.

if not exist "tools\python_sender.py" (
    echo [ERROR] tools\python_sender.py not found.
    echo.
    pause
    exit /b 1
)
echo [OK] tools\python_sender.py found.

if not exist "backend\hybrid_cnn_extractor.h5" (
    echo [WARN] backend\hybrid_cnn_extractor.h5 missing.
    echo        Stage 2 will fall back to the legacy RF.
)
echo.

REM ---- start backend in a new window ----
echo [1/3] Starting backend in a new window...
start "PQD Backend" cmd /k "cd /d %CD% && venv312\Scripts\python.exe backend\app.py"

REM ---- wait for backend to be ready (up to 60 sec) ----
echo [2/3] Waiting for backend at http://127.0.0.1:5000 ...
set /a tries=0
:WAIT
set /a tries+=1
if %tries% GTR 60 (
    echo [WARN] Backend did not respond within 60s. Continuing anyway.
    goto AFTER_WAIT
)
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:5000/api/pipeline_status -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto WAIT
)
echo [OK] Backend responded after %tries% seconds.

:AFTER_WAIT

REM ---- open dashboard in default browser ----
echo [3/3] Opening dashboard at http://localhost:5000/dashboard ...
start "" "http://localhost:5000/dashboard"
timeout /t 1 /nobreak >nul

REM ---- start sender in a new window ----
echo [3/3] Starting Python sender in a new window...
start "PQD Sender" cmd /k "cd /d %CD% && venv312\Scripts\python.exe tools\python_sender.py"

echo.
echo ============================================================
echo  All started.
echo  - Backend window: titled "PQD Backend"
echo  - Sender window:  titled "PQD Sender"
echo  - Dashboard:      browser tab at /dashboard
echo.
echo  To stop: close both console windows (Ctrl-C in each).
echo ============================================================
echo.
echo (You can close this window now. The other two will keep running.)
pause
