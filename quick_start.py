#!/usr/bin/env python3
"""
Quick Start Script for Advanced Options Trading System
Bypasses dependency checks and starts the system directly
"""

import subprocess
import sys
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_backend():
    """Start the backend API"""
    logger.info("🚀 Starting Backend API...")
    try:
        subprocess.run([sys.executable, 'advanced_options_engine.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Backend failed: {e}")
    except KeyboardInterrupt:
        logger.info("Backend stopped")

def start_frontend():
    """Start the frontend dashboard"""
    logger.info("🎨 Starting Frontend Dashboard...")
    try:
        subprocess.run([sys.executable, 'advanced_dashboard.py'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend failed: {e}")
    except KeyboardInterrupt:
        logger.info("Frontend stopped")

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("🚀 ADVANCED OPTIONS TRADING SYSTEM - QUICK START")
    logger.info("=" * 60)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend in a separate thread
    frontend_thread = threading.Thread(target=start_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    logger.info("✅ System started!")
    logger.info("🌐 Dashboard: http://localhost:8050")
    logger.info("🔧 API: http://localhost:8001")
    logger.info("📊 API Docs: http://localhost:8001/docs")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Stopping system...")
        sys.exit(0)

if __name__ == "__main__":
    main()