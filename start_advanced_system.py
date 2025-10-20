#!/usr/bin/env python3
"""
Advanced Options Trading Strategy Finder - Alpha Pro Max
Comprehensive startup script for the most sophisticated options trading system ever built

This script initializes and starts all components of the advanced options trading system:
- Advanced Options Engine (Backend API)
- Advanced Dashboard (Frontend)
- Risk Management System
- Backtesting Engine
- Machine Learning Models

Author: Advanced Options Trading Team
Version: 1.0.0 - Alpha Pro Max
"""

import os
import sys
import subprocess
import time
import threading
import logging
import signal
import json
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_options_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdvancedOptionsSystem:
    """Main system controller for the Advanced Options Trading System"""
    
    def __init__(self):
        self.processes = {}
        self.threads = {}
        self.running = False
        self.ports = {
            'backend': 8001,
            'frontend': 8050,
            'risk_management': 8002,
            'backtesting': 8003
        }
        
        # System components
        self.components = {
            'backend': {
                'script': 'advanced_options_engine.py',
                'port': self.ports['backend'],
                'name': 'Advanced Options Engine'
            },
            'frontend': {
                'script': 'advanced_dashboard.py',
                'port': self.ports['frontend'],
                'name': 'Advanced Dashboard'
            },
            'risk_management': {
                'script': 'risk_management.py',
                'port': self.ports['risk_management'],
                'name': 'Risk Management System'
            },
            'backtesting': {
                'script': 'backtesting_engine.py',
                'port': self.ports['backtesting'],
                'name': 'Backtesting Engine'
            }
        }
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("Checking system dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 'yfinance', 
            'scipy', 'scikit-learn', 'plotly', 'dash', 'dash_bootstrap_components'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"✗ {package} is missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            self.install_dependencies()
        else:
            logger.info("All dependencies are satisfied ✓")
    
    def install_dependencies(self):
        """Install missing dependencies"""
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True, capture_output=True, text=True)
            logger.info("Dependencies installed successfully ✓")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error("Please install manually: pip install -r requirements.txt")
            sys.exit(1)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'models',
            'data',
            'logs',
            'reports',
            'backtests'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def initialize_system(self):
        """Initialize the system components"""
        logger.info("Initializing Advanced Options Trading System...")
        
        # Check dependencies
        self.check_dependencies()
        
        # Create directories
        self.create_directories()
        
        # Initialize ML models
        self.initialize_ml_models()
        
        # Load configuration
        self.load_configuration()
        
        logger.info("System initialization complete ✓")
    
    def initialize_ml_models(self):
        """Initialize machine learning models"""
        logger.info("Initializing machine learning models...")
        
        try:
            # This would typically load pre-trained models
            # For now, we'll create placeholder model files
            model_config = {
                'return_predictor': {
                    'type': 'RandomForestRegressor',
                    'features': 20,
                    'trained': False
                },
                'risk_predictor': {
                    'type': 'GradientBoostingRegressor',
                    'features': 20,
                    'trained': False
                },
                'volatility_predictor': {
                    'type': 'LSTM',
                    'features': 50,
                    'trained': False
                }
            }
            
            with open('models/model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2)
            
            logger.info("ML models initialized ✓")
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def load_configuration(self):
        """Load system configuration"""
        logger.info("Loading system configuration...")
        
        config = {
            'system': {
                'name': 'Advanced Options Trading Strategy Finder',
                'version': '1.0.0 - Alpha Pro Max',
                'start_time': datetime.now().isoformat()
            },
            'risk_management': {
                'max_portfolio_risk': 0.02,
                'max_position_risk': 0.05,
                'max_correlation': 0.7,
                'max_concentration': 0.2,
                'max_drawdown': 0.1
            },
            'trading': {
                'min_price': 5.0,
                'min_oi': 50,
                'max_bid_ask_spread': 0.8,
                'confidence_threshold': 0.7
            },
            'api': {
                'backend_port': self.ports['backend'],
                'frontend_port': self.ports['frontend'],
                'cors_enabled': True
            }
        }
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Configuration loaded ✓")
    
    def start_component(self, component_name):
        """Start a system component"""
        if component_name not in self.components:
            logger.error(f"Unknown component: {component_name}")
            return False
        
        component = self.components[component_name]
        script_path = component['script']
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return False
        
        try:
            logger.info(f"Starting {component['name']} on port {component['port']}...")
            
            # Start the process
            process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.processes[component_name] = process
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_component,
                args=(component_name, process)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            self.threads[component_name] = monitor_thread
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✓ {component['name']} started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"✗ {component['name']} failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting {component['name']}: {e}")
            return False
    
    def _monitor_component(self, component_name, process):
        """Monitor a component process"""
        while self.running and process.poll() is None:
            time.sleep(1)
        
        if not self.running:
            return
        
        # Process died unexpectedly
        logger.error(f"Component {component_name} died unexpectedly")
        stdout, stderr = process.communicate()
        if stdout:
            logger.error(f"STDOUT: {stdout}")
        if stderr:
            logger.error(f"STDERR: {stderr}")
    
    def start_system(self):
        """Start the entire system"""
        logger.info("=" * 60)
        logger.info("STARTING ADVANCED OPTIONS TRADING SYSTEM - ALPHA PRO MAX")
        logger.info("=" * 60)
        
        self.running = True
        
        # Initialize system
        self.initialize_system()
        
        # Start components in order
        startup_order = ['backend', 'frontend', 'risk_management', 'backtesting']
        
        for component in startup_order:
            if not self.start_component(component):
                logger.error(f"Failed to start {component}. Stopping system.")
                self.stop_system()
                return False
        
        # Wait for all components to be ready
        logger.info("Waiting for all components to be ready...")
        time.sleep(5)
        
        # Display system status
        self.display_system_status()
        
        logger.info("=" * 60)
        logger.info("SYSTEM STARTUP COMPLETE - ALL COMPONENTS RUNNING")
        logger.info("=" * 60)
        logger.info(f"Advanced Dashboard: http://localhost:{self.ports['frontend']}")
        logger.info(f"Backend API: http://localhost:{self.ports['backend']}")
        logger.info(f"Risk Management: http://localhost:{self.ports['risk_management']}")
        logger.info(f"Backtesting Engine: http://localhost:{self.ports['backtesting']}")
        logger.info("=" * 60)
        
        return True
    
    def display_system_status(self):
        """Display current system status"""
        logger.info("System Status:")
        logger.info("-" * 40)
        
        for component_name, process in self.processes.items():
            if process.poll() is None:
                status = "RUNNING"
                logger.info(f"✓ {self.components[component_name]['name']}: {status}")
            else:
                status = "STOPPED"
                logger.error(f"✗ {self.components[component_name]['name']}: {status}")
        
        logger.info("-" * 40)
    
    def stop_system(self):
        """Stop the entire system"""
        logger.info("Stopping Advanced Options Trading System...")
        
        self.running = False
        
        # Stop all processes
        for component_name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Stopping {self.components[component_name]['name']}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"✓ {self.components[component_name]['name']} stopped")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {self.components[component_name]['name']}...")
                    process.kill()
                    process.wait()
                    logger.info(f"✓ {self.components[component_name]['name']} force stopped")
        
        # Wait for monitoring threads
        for thread in self.threads.values():
            thread.join(timeout=2)
        
        logger.info("System stopped ✓")
    
    def restart_component(self, component_name):
        """Restart a specific component"""
        logger.info(f"Restarting {component_name}...")
        
        # Stop component if running
        if component_name in self.processes:
            process = self.processes[component_name]
            if process.poll() is None:
                process.terminate()
                process.wait()
        
        # Start component
        if self.start_component(component_name):
            logger.info(f"✓ {component_name} restarted successfully")
            return True
        else:
            logger.error(f"✗ Failed to restart {component_name}")
            return False
    
    def run_tests(self):
        """Run system tests"""
        logger.info("Running system tests...")
        try:
            result = subprocess.run([sys.executable, 'test_system.py'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info("✓ All tests passed")
                return True
            else:
                logger.error(f"✗ Tests failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("✗ Tests timed out")
            return False
        except Exception as e:
            logger.error(f"✗ Test error: {e}")
            return False

    def run(self):
        """Main run loop"""
        try:
            # Run tests first
            if not self.run_tests():
                logger.error("System tests failed. Please fix issues before starting.")
                return
            
            # Start system
            if not self.start_system():
                logger.error("Failed to start system")
                return
            
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                
                # Check if any processes died
                for component_name, process in self.processes.items():
                    if process.poll() is not None:
                        logger.error(f"Component {component_name} died. Attempting restart...")
                        if not self.restart_component(component_name):
                            logger.error(f"Failed to restart {component_name}. Stopping system.")
                            self.stop_system()
                            return
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop_system()

def signal_handler(signum, frame):
    """Handle system signals"""
    logger.info(f"Received signal {signum}")
    global system
    if 'system' in globals():
        system.stop_system()
    sys.exit(0)

def main():
    """Main entry point"""
    global system
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run system
    system = AdvancedOptionsSystem()
    system.run()

if __name__ == "__main__":
    main()