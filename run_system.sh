#!/bin/bash

# Advanced Options Trading Strategy Finder - Quick Run Script
# This script provides easy commands to run the system

echo "🚀 Advanced Options Trading Strategy Finder - Alpha Pro Max"
echo "=========================================================="

# Function to check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.8"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        echo "✅ Python $PYTHON_VERSION detected"
    else
        echo "❌ Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Function to install dependencies
install_deps() {
    echo "📦 Installing dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    echo "🧪 Running system tests..."
    $PYTHON_CMD test_system.py
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed"
    else
        echo "❌ Tests failed. Please check the errors above."
        exit 1
    fi
}

# Function to start system
start_system() {
    echo "🚀 Starting Advanced Options Trading System..."
    echo "Dashboard will be available at: http://localhost:8050"
    echo "API will be available at: http://localhost:8001"
    echo "Press Ctrl+C to stop the system"
    echo ""
    $PYTHON_CMD start_advanced_system.py
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install    Install dependencies"
    echo "  test       Run system tests"
    echo "  start      Start the system"
    echo "  full       Install, test, and start (recommended)"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 full          # Complete setup and start"
    echo "  $0 install       # Install dependencies only"
    echo "  $0 test          # Run tests only"
    echo "  $0 start         # Start system (assumes setup is complete)"
}

# Main script logic
case "${1:-full}" in
    "install")
        check_python
        install_deps
        ;;
    "test")
        check_python
        run_tests
        ;;
    "start")
        check_python
        start_system
        ;;
    "full")
        check_python
        install_deps
        run_tests
        start_system
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_help
        exit 1
        ;;
esac