@echo off
echo ♟️ ChessBot - Launching Chess Game
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Try to launch the game launcher
echo 🚀 Starting ChessBot...
python launch_game.py

REM If launcher fails, try direct launch
if errorlevel 1 (
    echo.
    echo 🔄 Trying direct launch...
    python play_chess.py
)

echo.
echo 👋 Thanks for playing ChessBot!
pause
