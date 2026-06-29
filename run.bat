@echo off
echo ============================================================
echo   AI-Based Helmet Detection System - Quick Launch
echo ============================================================
echo.

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No virtual environment found.
    echo [INFO] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

echo.
echo [INFO] Installing/updating dependencies...
pip install -r requirements.txt --quiet

echo.
echo ============================================================
echo   Choose a mode:
echo   1. Streamlit Web App (recommended for demos)
echo   2. Webcam Detection (Standard Fast Model)
echo   3. Webcam Detection (High Accuracy Medium Model + Tracking)
echo   4. Image Detection (CLI)
echo   5. Video Detection (with Tracking)
echo   6. Fix False Positives (Automated Webcam Training)
echo ============================================================
echo.

set /p choice="Enter your choice (1/2/3/4/5/6): "

if "%choice%"=="1" (
    echo.
    echo [INFO] Launching Streamlit app...
    streamlit run src/app.py
) else if "%choice%"=="2" (
    echo.
    echo [INFO] Starting standard webcam detection...
    python -m src.detect --webcam
) else if "%choice%"=="3" (
    echo.
    echo [INFO] Starting High Accuracy webcam detection with tracking...
    python -m src.detect --webcam --track --model models/best.pt --half --confidence 0.5
) else if "%choice%"=="4" (
    set /p imgpath="Enter image path: "
    echo.
    echo [INFO] Running detection on image...
    python -m src.detect --image "%imgpath%"
) else if "%choice%"=="5" (
    set /p vidpath="Enter video path: "
    echo.
    echo [INFO] Running detection on video with tracking...
    python -m src.detect --video "%vidpath%" --track
) else if "%choice%"=="6" (
    echo.
    echo [INFO] Launching Automated False Positive Fixer...
    python easy_false_positive_fix.py
) else (
    echo [ERROR] Invalid choice. Please run again.
)

pause
