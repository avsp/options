# 🚀 Advanced Options Trading Strategy Finder - Setup & Run Guide

## 📋 Prerequisites

Before running the system, ensure you have:

- **Python 3.8 or higher** (3.9+ recommended)
- **8GB RAM minimum** (16GB recommended for optimal performance)
- **10GB free disk space**
- **Internet connection** (for real-time market data)
- **Git** (for cloning the repository)

## 🔧 Installation Steps

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-options-trading-system.git

# Navigate to the project directory
cd advanced-options-trading-system
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually if needed:
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install pandas==2.1.3
pip install numpy==1.24.3
pip install yfinance==0.2.18
pip install scipy==1.11.4
pip install scikit-learn==1.3.2
pip install plotly==5.17.0
pip install dash==2.14.2
pip install dash-bootstrap-components==1.5.0
```

### 4. Verify Installation

```bash
# Run the test suite to verify everything is working
python test_system.py
```

You should see output like:
```
==================================================
RUNNING ADVANCED OPTIONS SYSTEM TESTS
==================================================

Running Import Test...
✓ All imports successful
✓ Import Test PASSED

Running Basic Functionality...
✓ Pandas working
✓ NumPy working
✓ YFinance working
✓ Basic Functionality PASSED

... (more tests)

==================================================
TEST RESULTS: 8/8 tests passed
==================================================
🎉 ALL TESTS PASSED! System is ready to use.
```

## 🚀 Running the System

### Option 1: Automated Startup (Recommended)

```bash
# Start the entire system with one command
python start_advanced_system.py
```

This will:
- Check all dependencies
- Initialize the system
- Start all components (Backend, Frontend, Risk Management, Backtesting)
- Display system status
- Provide access URLs

### Option 2: Manual Component Startup

If you prefer to start components individually:

#### Start Backend API
```bash
python advanced_options_engine.py
```

#### Start Frontend Dashboard (in new terminal)
```bash
python advanced_dashboard.py
```

#### Start Risk Management (in new terminal)
```bash
python risk_management.py
```

#### Start Backtesting Engine (in new terminal)
```bash
python backtesting_engine.py
```

## 🌐 Accessing the System

Once the system is running, you can access:

### Main Dashboard
- **URL**: http://localhost:8050
- **Description**: Main user interface with strategy visualization and controls

### Backend API
- **URL**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **Description**: RESTful API for programmatic access

### Risk Management
- **URL**: http://localhost:8002
- **Description**: Risk monitoring and management interface

### Backtesting Engine
- **URL**: http://localhost:8003
- **Description**: Historical analysis and backtesting interface

## 📊 Using the System

### 1. Basic Usage

1. **Open the Dashboard**: Navigate to http://localhost:8050
2. **Configure Universe**: Click "Edit Universe" to add/remove symbols
3. **Set Parameters**: Adjust risk parameters and strategy preferences
4. **Run Scan**: Click "Run Advanced Scan" to generate strategies
5. **Analyze Results**: Review generated portfolios and individual strategies

### 2. Advanced Usage

#### API Integration
```python
import requests

# Get market regime
response = requests.get('http://localhost:8001/market-regime')
market_data = response.json()

# Run strategy scan
scan_request = {
    'max_risk': 0.02,
    'min_return': 0.15,
    'strategy_types': ['income', 'directional'],
    'max_positions': 10
}
response = requests.post('http://localhost:8001/scan', json=scan_request)
strategies = response.json()
```

#### Custom Configuration
Edit `config.json` to customize system parameters:
```json
{
    "risk_management": {
        "max_portfolio_risk": 0.02,
        "max_position_risk": 0.05,
        "max_correlation": 0.7
    },
    "trading": {
        "min_price": 5.0,
        "min_oi": 50,
        "confidence_threshold": 0.7
    }
}
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Find and kill processes using the ports
# For port 8001 (Backend):
lsof -ti:8001 | xargs kill -9

# For port 8050 (Frontend):
lsof -ti:8050 | xargs kill -9

# Or change ports in the configuration
```

#### 2. Import Errors
**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install missing package specifically
pip install package_name
```

#### 3. Memory Issues
**Error**: `MemoryError` or slow performance

**Solution**:
- Reduce symbol universe size
- Close other applications
- Increase system RAM
- Adjust scan parameters

#### 4. Data Connection Issues
**Error**: `ConnectionError` or no data

**Solution**:
- Check internet connection
- Verify Yahoo Finance is accessible
- Try different symbols
- Check market hours

#### 5. System Won't Start
**Error**: Various startup errors

**Solution**:
```bash
# Run tests first
python test_system.py

# Check Python version
python --version

# Verify all files are present
ls -la

# Check logs
tail -f advanced_options_system.log
```

### Log Files

The system creates several log files for debugging:

- **System Log**: `advanced_options_system.log`
- **Error Log**: `logs/error.log`
- **Performance Log**: `logs/performance.log`

### Getting Help

1. **Check Logs**: Review log files for error details
2. **Run Tests**: Use `python test_system.py` to verify system health
3. **Check Documentation**: Review README.md and strategy_guide.md
4. **GitHub Issues**: Create an issue on the GitHub repository

## 🔧 Configuration Options

### System Configuration

Edit `config.json` to customize:

```json
{
    "system": {
        "name": "Advanced Options Trading Strategy Finder",
        "version": "1.0.0 - Alpha Pro Max"
    },
    "risk_management": {
        "max_portfolio_risk": 0.02,
        "max_position_risk": 0.05,
        "max_correlation": 0.7,
        "max_concentration": 0.2,
        "max_drawdown": 0.1
    },
    "trading": {
        "min_price": 5.0,
        "min_oi": 50,
        "max_bid_ask_spread": 0.8,
        "confidence_threshold": 0.7
    },
    "api": {
        "backend_port": 8001,
        "frontend_port": 8050,
        "cors_enabled": true
    }
}
```

### Strategy Configuration

Customize strategy parameters in `advanced_options_engine.py`:

```python
# Example: Custom Iron Condor parameters
IRON_CONDOR_CONFIG = {
    'max_width': 5.0,
    'min_credit': 0.5,
    'dte_range': [30, 45],
    'delta_target': 0.16
}
```

## 📈 Performance Optimization

### For Better Performance

1. **Increase RAM**: 16GB+ recommended
2. **Use SSD**: Faster disk access
3. **Close Other Apps**: Free up system resources
4. **Reduce Universe**: Limit symbol count
5. **Adjust Scan Frequency**: Reduce real-time updates

### For Production Use

1. **Use Database**: Replace file storage with database
2. **Add Authentication**: Implement user authentication
3. **Enable HTTPS**: Secure API endpoints
4. **Add Monitoring**: Implement system monitoring
5. **Scale Horizontally**: Use load balancers

## 🚀 Deployment Options

### Local Development
```bash
python start_advanced_system.py
```

### Docker Deployment
```dockerfile
# Create Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8001 8050
CMD ["python", "start_advanced_system.py"]
```

### Cloud Deployment
- **AWS**: Use EC2 with load balancer
- **Google Cloud**: Use Compute Engine
- **Azure**: Use Virtual Machines
- **Heroku**: Use Procfile for deployment

## 📚 Additional Resources

### Documentation
- **README.md**: Main documentation
- **strategy_guide.md**: Comprehensive strategy guide
- **SYSTEM_OVERVIEW.md**: Technical architecture
- **API Documentation**: http://localhost:8001/docs

### Video Tutorials
- System Overview
- Basic Usage
- Advanced Features
- Risk Management

### Community
- GitHub Issues
- Discussion Forums
- Discord Server
- Email Support

## 🎯 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Tests passed
- [ ] System started
- [ ] Dashboard accessible
- [ ] First scan completed

## 🆘 Emergency Procedures

### System Crashes
1. Check logs for errors
2. Restart system
3. Verify dependencies
4. Contact support

### Data Loss
1. Check backup files
2. Restore from git
3. Re-run initialization
4. Contact support

### Performance Issues
1. Check system resources
2. Reduce scan parameters
3. Restart components
4. Contact support

---

**🎉 Congratulations! You now have the most advanced options trading system ever built!**

**Start discovering profitable strategies today!** 🚀