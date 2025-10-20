@echo off
REM Advanced Options Trading Strategy Finder - Quick Run Script for Windows
REM This script provides easy commands to run the system

echo 🚀 Advanced Options Trading Strategy Finder - Alpha Pro Max
echo ==========================================================

REM Function to check if Python is installed
:check_python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python detected
goto :eof

REM Function to install dependencies
:install_deps
echo 📦 Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully
goto :eof

REM Function to run tests
:run_tests
echo 🧪 Running system tests...
python test_system.py
if %errorlevel% neq 0 (
    echo ❌ Tests failed. Please check the errors above.
    pause
    exit /b 1
)
echo ✅ All tests passed
goto :eof

REM Function to start system
:start_system
echo 🚀 Starting Advanced Options Trading System...
echo Dashboard will be available at: http://localhost:8050
echo API will be available at: http://localhost:8001
echo Press Ctrl+C to stop the system
echo.
python start_advanced_system.py
goto :eof

REM Function to show help
:show_help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   install    Install dependencies
echo   test       Run system tests
echo   start      Start the system
echo   full       Install, test, and start (recommended)
echo   help       Show this help message
echo.
echo Examples:
echo   %0 full          # Complete setup and start
echo   %0 install       # Install dependencies only
echo   %0 test          # Run tests only
echo   %0 start         # Start system (assumes setup is complete)
goto :eof

REM Main script logic
if "%1"=="install" (
    call :check_python
    call :install_deps
    goto :eof
)
if "%1"=="test" (
    call :check_python
    call :run_tests
    goto :eof
)
if "%1"=="start" (
    call :check_python
    call :start_system
    goto :eof
)
if "%1"=="help" (
    call :show_help
    goto :eof
)
if "%1"=="-h" (
    call :show_help
    goto :eof
)
if "%1"=="--help" (
    call :show_help
    goto :eof
)
if "%1"=="" (
    call :check_python
    call :install_deps
    call :run_tests
    call :start_system
    goto :eof
)

echo ❌ Unknown command: %1
call :show_help
pause
exit /b 1