#!/usr/bin/env python3
"""
Simple test to verify the system components work
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports"""
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from scipy import stats
        from sklearn.ensemble import RandomForestRegressor
        import plotly.graph_objects as go
        import dash
        import fastapi
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        
        # Test pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        
        # Test numpy
        arr = np.array([1, 2, 3, 4, 5])
        assert np.mean(arr) == 3.0
        
        # Test yfinance
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        assert 'regularMarketPrice' in info
        
        logger.info("✅ Basic functionality working")
        return True
    except Exception as e:
        logger.error(f"❌ Basic functionality error: {e}")
        return False

def test_black_scholes():
    """Test Black-Scholes calculation"""
    try:
        from advanced_options_engine import black_scholes
        
        result = black_scholes('call', 100, 100, 0.25, 0.05, 0.2)
        assert 'price' in result
        assert 'delta' in result
        
        logger.info("✅ Black-Scholes working")
        return True
    except Exception as e:
        logger.error(f"❌ Black-Scholes error: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Running Simple System Tests")
    logger.info("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Black-Scholes", test_black_scholes)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        return True
    else:
        logger.error("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)