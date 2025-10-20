#!/usr/bin/env python3
"""
Test script for Advanced Options Trading Strategy Finder
This script tests the core functionality of the system
"""

import sys
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from scipy import stats
        from sklearn.ensemble import RandomForestRegressor
        import plotly.graph_objects as go
        import dash
        import fastapi
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    logger.info("Testing basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        
        # Test pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        logger.info("✓ Pandas working")
        
        # Test numpy
        arr = np.array([1, 2, 3, 4, 5])
        assert np.mean(arr) == 3.0
        logger.info("✓ NumPy working")
        
        # Test yfinance
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        assert 'regularMarketPrice' in info
        logger.info("✓ YFinance working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        return False

def test_black_scholes():
    """Test Black-Scholes calculation"""
    logger.info("Testing Black-Scholes calculation...")
    
    try:
        from advanced_options_engine import black_scholes
        
        # Test call option
        result = black_scholes('call', 100, 100, 0.25, 0.05, 0.2)
        assert 'price' in result
        assert 'delta' in result
        assert 'gamma' in result
        assert 'theta' in result
        assert 'vega' in result
        assert 'rho' in result
        logger.info("✓ Black-Scholes calculation working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Black-Scholes test failed: {e}")
        return False

def test_market_analyzer():
    """Test market analyzer"""
    logger.info("Testing market analyzer...")
    
    try:
        from advanced_options_engine import MarketAnalyzer
        
        analyzer = MarketAnalyzer()
        logger.info("✓ Market analyzer created")
        
        return True
    except Exception as e:
        logger.error(f"✗ Market analyzer test failed: {e}")
        return False

def test_strategy_generator():
    """Test strategy generator"""
    logger.info("Testing strategy generator...")
    
    try:
        from advanced_options_engine import AdvancedStrategyGenerator
        
        generator = AdvancedStrategyGenerator()
        assert len(generator.strategy_templates) > 0
        logger.info("✓ Strategy generator working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Strategy generator test failed: {e}")
        return False

def test_risk_manager():
    """Test risk manager"""
    logger.info("Testing risk manager...")
    
    try:
        from risk_management import RiskManager
        
        risk_manager = RiskManager()
        assert risk_manager.max_portfolio_risk == 0.02
        logger.info("✓ Risk manager working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Risk manager test failed: {e}")
        return False

def test_backtester():
    """Test backtester"""
    logger.info("Testing backtester...")
    
    try:
        from backtesting_engine import OptionsBacktester
        
        backtester = OptionsBacktester()
        assert backtester.initial_capital == 100000
        logger.info("✓ Backtester working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Backtester test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard"""
    logger.info("Testing dashboard...")
    
    try:
        from advanced_dashboard import app
        
        assert app is not None
        logger.info("✓ Dashboard working")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dashboard test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("RUNNING ADVANCED OPTIONS SYSTEM TESTS")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Black-Scholes", test_black_scholes),
        ("Market Analyzer", test_market_analyzer),
        ("Strategy Generator", test_strategy_generator),
        ("Risk Manager", test_risk_manager),
        ("Backtester", test_backtester),
        ("Dashboard", test_dashboard)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info("=" * 50)
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System is ready to use.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)