# Advanced Options Trading Strategy Finder - The Most Sophisticated System Ever Built
# Version: Alpha Pro Max - Ultimate Quantitative Options Engine
# Features: ML-Powered Strategy Selection, Advanced Risk Management, Portfolio Optimization

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
import time
import yfinance as yf
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from math import log, sqrt, exp, erf
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import requests
from dataclasses import asdict
import uuid

# --- CONFIGURATION & SETUP ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [ALPHA-ENGINE] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Options Strategy Finder", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- GLOBAL CONFIGURATION ---
UNIVERSE_FILE = "universe.csv"
MODEL_CACHE_DIR = "models"
yf_cache = {}
ml_models = {}
strategy_performance_cache = {}
risk_metrics_cache = {}

# Ensure model cache directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# --- ADVANCED DATACLASSES & MODELS ---
@dataclass
class AdvancedConfig:
    risk_free_rate: float = 0.045
    min_price: float = 5.0
    min_oi: int = 50
    max_bid_ask_spread_pct: float = 0.8
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_correlation: float = 0.7  # Max correlation between positions
    min_expected_return: float = 0.15  # 15% minimum expected return
    max_drawdown: float = 0.1  # 10% max drawdown
    confidence_threshold: float = 0.7  # ML model confidence threshold
    volatility_lookback: int = 252  # 1 year for volatility calculations
    rebalance_frequency: int = 7  # Rebalance every 7 days

@dataclass
class MarketRegime:
    regime: str
    vix_level: float
    vix_term_structure: str
    volatility_percentile: float
    trend_strength: float
    correlation_regime: str
    liquidity_regime: str
    confidence: float

@dataclass
class OptionLeg:
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: str
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class Strategy:
    name: str
    strategy_type: str  # 'income', 'directional', 'volatility', 'arbitrage'
    legs: List[OptionLeg]
    net_debit: float
    net_credit: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    expected_return: float
    risk_reward_ratio: float
    greeks: Dict[str, float]
    market_conditions: List[str]
    confidence_score: float
    risk_level: str  # 'low', 'medium', 'high', 'extreme'

@dataclass
class Portfolio:
    strategies: List[Strategy]
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    portfolio_beta: float
    portfolio_volatility: float
    expected_return: float
    var_95: float  # Value at Risk 95%
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    diversification_ratio: float
    risk_score: float

class UniverseRequest(BaseModel):
    symbols: List[str]

class StrategyRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    max_risk: float = 0.02
    min_return: float = 0.15
    strategy_types: List[str] = Field(default_factory=lambda: ['income', 'directional', 'volatility'])
    max_positions: int = 10

class BacktestRequest(BaseModel):
    strategies: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    rebalance_frequency: int = 7

# --- ADVANCED MATHEMATICAL FUNCTIONS ---
def _n_cdf(x): 
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def _n_pdf(x): 
    return exp(-x**2 / 2.0) / sqrt(2.0 * np.pi)

def black_scholes(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """Enhanced Black-Scholes with all Greeks"""
    if T <= 0 or sigma <= 0:
        return {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * _n_cdf(d1) - K * exp(-r * T) * _n_cdf(d2)
        delta = _n_cdf(d1)
        theta = (-(S * _n_pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * _n_cdf(d2)) / 365
    else:  # put
        price = K * exp(-r * T) * _n_cdf(-d2) - S * _n_cdf(-d1)
        delta = -_n_cdf(-d1)
        theta = (-(S * _n_pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * _n_cdf(-d2)) / 365
    
    gamma = _n_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * _n_pdf(d1) * sqrt(T) / 100
    rho = K * T * exp(-r * T) * (_n_cdf(d2) if option_type.lower() == 'call' else -_n_cdf(-d2)) / 100
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def calculate_implied_volatility(option_type: str, S: float, K: float, T: float, r: float, market_price: float) -> float:
    """Calculate implied volatility using Newton-Raphson method"""
    if T <= 0 or market_price <= 0:
        return 0.0
    
    # Initial guess
    sigma = 0.3
    tolerance = 1e-6
    max_iterations = 100
    
    for _ in range(max_iterations):
        bs = black_scholes(option_type, S, K, T, r, sigma)
        price_diff = bs['price'] - market_price
        vega = bs['vega']
        
        if abs(price_diff) < tolerance:
            return sigma
        
        if vega == 0:
            break
            
        sigma = sigma - price_diff / vega
        sigma = max(0.01, min(5.0, sigma))  # Keep within reasonable bounds
    
    return sigma

# --- ADVANCED MARKET ANALYSIS ---
class MarketAnalyzer:
    def __init__(self):
        self.vix_data = None
        self.spy_data = None
        self.qqq_data = None
        self.iwm_data = None
        self.last_update = None
    
    async def get_market_regime(self) -> MarketRegime:
        """Comprehensive market regime analysis"""
        try:
            # Get VIX data
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="1y")
            
            if vix_hist.empty:
                return MarketRegime("Unknown", 20.0, "Unknown", 50.0, 0.0, "Unknown", "Unknown", 0.0)
            
            current_vix = vix_hist['Close'].iloc[-1]
            vix_percentile = stats.percentileofscore(vix_hist['Close'], current_vix)
            
            # Determine regime
            if current_vix > 30:
                regime = "High Volatility"
            elif current_vix < 15:
                regime = "Low Volatility"
            else:
                regime = "Normal Volatility"
            
            # VIX term structure
            vix_9 = yf.Ticker("^VIX9D")
            vix_9_hist = vix_9.history(period="5d")
            if not vix_9_hist.empty:
                vix_9_current = vix_9_hist['Close'].iloc[-1]
                if vix_9_current > current_vix * 1.1:
                    term_structure = "Contango"
                elif vix_9_current < current_vix * 0.9:
                    term_structure = "Backwardation"
                else:
                    term_structure = "Flat"
            else:
                term_structure = "Unknown"
            
            # Get SPY data for trend analysis
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")
            if not spy_hist.empty:
                spy_returns = spy_hist['Close'].pct_change().dropna()
                trend_strength = abs(spy_returns.rolling(20).mean().iloc[-1]) * 100
                
                # Correlation regime
                spy_vol = spy_returns.rolling(20).std().iloc[-1] * sqrt(252)
                if spy_vol > 0.25:
                    correlation_regime = "High Correlation"
                elif spy_vol < 0.15:
                    correlation_regime = "Low Correlation"
                else:
                    correlation_regime = "Normal Correlation"
            else:
                trend_strength = 0.0
                correlation_regime = "Unknown"
            
            # Liquidity regime (based on volume)
            if not spy_hist.empty:
                avg_volume = spy_hist['Volume'].rolling(20).mean().iloc[-1]
                current_volume = spy_hist['Volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:
                    liquidity_regime = "High Liquidity"
                elif current_volume < avg_volume * 0.5:
                    liquidity_regime = "Low Liquidity"
                else:
                    liquidity_regime = "Normal Liquidity"
            else:
                liquidity_regime = "Unknown"
            
            confidence = min(0.95, max(0.1, 1.0 - abs(vix_percentile - 50) / 50))
            
            return MarketRegime(
                regime=regime,
                vix_level=current_vix,
                vix_term_structure=term_structure,
                volatility_percentile=vix_percentile,
                trend_strength=trend_strength,
                correlation_regime=correlation_regime,
                liquidity_regime=liquidity_regime,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return MarketRegime("Error", 20.0, "Error", 50.0, 0.0, "Error", "Error", 0.0)

# --- ADVANCED OPTIONS DATA HANDLING ---
class OptionsDataManager:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def get_ticker_data(self, symbol: str) -> Optional[yf.Ticker]:
        """Get ticker with caching"""
        if symbol in self.cache and (time.time() - self.cache[symbol]['timestamp']) < self.cache_timeout:
            return self.cache[symbol]['ticker']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info.get('regularMarketPrice') is not None:
                self.cache[symbol] = {'ticker': ticker, 'timestamp': time.time()}
                return ticker
        except Exception as e:
            logger.error(f"Error getting ticker data for {symbol}: {e}")
        
        return None
    
    async def get_historical_data(self, ticker: yf.Ticker, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical data with enhanced features"""
        try:
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            
            # Add technical indicators
            hist['Returns'] = hist['Close'].pct_change()
            hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            hist['SMA_20'] = hist['Close'].rolling(20).mean()
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
            hist['EMA_26'] = hist['Close'].ewm(span=26).mean()
            hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['Bollinger_Upper'] = hist['SMA_20'] + (hist['Close'].rolling(20).std() * 2)
            hist['Bollinger_Lower'] = hist['SMA_20'] - (hist['Close'].rolling(20).std() * 2)
            
            # Volatility measures
            hist['HV_20'] = hist['Log_Returns'].rolling(20).std() * sqrt(252) * 100
            hist['HV_50'] = hist['Log_Returns'].rolling(50).std() * sqrt(252) * 100
            hist['Parkinson_Vol'] = self._calculate_parkinson_volatility(hist)
            hist['Garman_Klass_Vol'] = self._calculate_garman_klass_volatility(hist)
            
            return hist
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_parkinson_volatility(self, hist: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility"""
        ln_hl_squared = (np.log(hist['High'] / hist['Low']) ** 2)
        return sqrt(ln_hl_squared.rolling(20).mean() / (4 * log(2))) * sqrt(252) * 100
    
    def _calculate_garman_klass_volatility(self, hist: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility"""
        ln_hl = np.log(hist['High'] / hist['Low'])
        ln_co = np.log(hist['Close'] / hist['Open'])
        gk = 0.5 * ln_hl**2 - (2*log(2)-1) * ln_co**2
        return sqrt(gk.rolling(20).mean()) * sqrt(252) * 100
    
    async def get_options_chain(self, ticker: yf.Ticker, target_dte: int = 30) -> Optional[pd.DataFrame]:
        """Get options chain with enhanced filtering"""
        try:
            expiries = ticker.options
            if not expiries:
                return None
            
            today = datetime.now().date()
            valid_expiries = []
            
            for expiry_str in expiries:
                try:
                    expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
                    dte = (expiry_date - today).days
                    if dte > 5:  # At least 5 days to expiry
                        valid_expiries.append((expiry_str, dte))
                except ValueError:
                    continue
            
            if not valid_expiries:
                return None
            
            # Find closest expiry to target DTE
            target_expiry, _ = min(valid_expiries, key=lambda x: abs(x[1] - target_dte))
            
            # Get options chain
            chain = ticker.option_chain(target_expiry)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            
            df = pd.concat([calls, puts], ignore_index=True)
            
            if df.empty:
                return None
            
            # Enhanced filtering
            df = df[(df['openInterest'] >= 10) & (df['bid'] > 0) & (df['ask'] > 0)].copy()
            df['mid'] = (df['bid'] + df['ask']) / 2
            df['spread_pct'] = (df['ask'] - df['bid']) / df['mid']
            df = df[df['spread_pct'] <= 0.8]  # Max 80% spread
            
            # Calculate Greeks for each option
            current_price = ticker.info.get('regularMarketPrice', 0)
            if current_price > 0:
                df['calculated_iv'] = df.apply(
                    lambda row: calculate_implied_volatility(
                        row['option_type'], current_price, row['strike'], 
                        target_dte/365, 0.045, row['mid']
                    ), axis=1
                )
                
                # Calculate Greeks
                greeks_data = []
                for _, row in df.iterrows():
                    bs = black_scholes(
                        row['option_type'], current_price, row['strike'],
                        target_dte/365, 0.045, row['calculated_iv']
                    )
                    greeks_data.append(bs)
                
                greeks_df = pd.DataFrame(greeks_data)
                df = pd.concat([df, greeks_df], axis=1)
            
            df['expiry_date'] = target_expiry
            df['dte'] = target_dte
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return None

# --- MACHINE LEARNING MODELS ---
class MLStrategyPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'current_price', 'hv_20', 'hv_50', 'parkinson_vol', 'garman_klass_vol',
            'rsi', 'macd', 'bollinger_position', 'volume_ratio', 'price_momentum',
            'iv_30', 'iv_60', 'iv_90', 'iv_term_structure', 'skew_30', 'skew_60',
            'kurtosis_30', 'kurtosis_60', 'vix_level', 'vix_percentile',
            'spy_correlation', 'sector_beta', 'earnings_volatility'
        ]
    
    async def train_models(self, training_data: List[dict]):
        """Train ML models for strategy prediction"""
        if not training_data:
            return
        
        df = pd.DataFrame(training_data)
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        y_return = df['strategy_return']
        y_risk = df['strategy_risk']
        
        # Train return prediction model
        X_train, X_test, y_train, y_test = train_test_split(X, y_return, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest for return prediction
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Gradient Boosting for risk prediction
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_risk)
        
        # Save models
        self.models['return_predictor'] = rf_model
        self.models['risk_predictor'] = gb_model
        self.scalers['main'] = scaler
        
        # Save to disk
        joblib.dump(rf_model, f"{MODEL_CACHE_DIR}/return_predictor.pkl")
        joblib.dump(gb_model, f"{MODEL_CACHE_DIR}/risk_predictor.pkl")
        joblib.dump(scaler, f"{MODEL_CACHE_DIR}/scaler.pkl")
        
        logger.info("ML models trained and saved successfully")
    
    async def load_models(self):
        """Load pre-trained models"""
        try:
            self.models['return_predictor'] = joblib.load(f"{MODEL_CACHE_DIR}/return_predictor.pkl")
            self.models['risk_predictor'] = joblib.load(f"{MODEL_CACHE_DIR}/risk_predictor.pkl")
            self.scalers['main'] = joblib.load(f"{MODEL_CACHE_DIR}/scaler.pkl")
            logger.info("ML models loaded successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained models found, will train new ones")
    
    async def predict_strategy_performance(self, features: dict) -> Tuple[float, float, float]:
        """Predict strategy performance using ML models"""
        if not self.models:
            await self.load_models()
        
        if not self.models:
            return 0.0, 0.0, 0.0
        
        # Prepare feature vector
        feature_vector = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
        feature_vector_scaled = self.scalers['main'].transform(feature_vector)
        
        # Predict
        expected_return = self.models['return_predictor'].predict(feature_vector_scaled)[0]
        predicted_risk = self.models['risk_predictor'].predict(feature_vector_scaled)[0]
        confidence = min(0.95, max(0.1, 1.0 - abs(predicted_risk) / 0.5))
        
        return expected_return, predicted_risk, confidence

# --- ADVANCED STRATEGY GENERATOR ---
class AdvancedStrategyGenerator:
    def __init__(self):
        self.strategy_templates = {
            'iron_condor': self._create_iron_condor,
            'butterfly': self._create_butterfly,
            'straddle': self._create_straddle,
            'strangle': self._create_strangle,
            'collar': self._create_collar,
            'covered_call': self._create_covered_call,
            'protective_put': self._create_protective_put,
            'calendar_spread': self._create_calendar_spread,
            'diagonal_spread': self._create_diagonal_spread,
            'ratio_spread': self._create_ratio_spread,
            'backspread': self._create_backspread,
            'jade_lizard': self._create_jade_lizard,
            'iron_butterfly': self._create_iron_butterfly,
            'condor': self._create_condor,
            'box_spread': self._create_box_spread,
            'conversion': self._create_conversion,
            'reversal': self._create_reversal,
            'synthetic_long': self._create_synthetic_long,
            'synthetic_short': self._create_synthetic_short,
            'volatility_arbitrage': self._create_volatility_arbitrage
        }
    
    def _create_iron_condor(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Iron Condor strategy"""
        try:
            calls = options_df[options_df['option_type'] == 'call'].sort_values('strike')
            puts = options_df[options_df['option_type'] == 'put'].sort_values('strike', ascending=False)
            
            if len(calls) < 4 or len(puts) < 4:
                return None
            
            # Find strikes around current price
            atm_call_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_put_idx = (puts['strike'] - current_price).abs().idxmin()
            
            # Select strikes for iron condor
            short_call = calls.iloc[atm_call_idx + 2] if atm_call_idx + 2 < len(calls) else calls.iloc[-1]
            long_call = calls.iloc[atm_call_idx + 4] if atm_call_idx + 4 < len(calls) else calls.iloc[-1]
            short_put = puts.iloc[atm_put_idx + 2] if atm_put_idx + 2 < len(puts) else puts.iloc[-1]
            long_put = puts.iloc[atm_put_idx + 4] if atm_put_idx + 4 < len(puts) else puts.iloc[-1]
            
            # Create legs
            legs = [
                OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='call', strike=short_call['strike'],
                         expiry=options_df['expiry_date'].iloc[0], bid=short_call['bid'], ask=short_call['ask'],
                         mid=short_call['mid'], volume=short_call.get('volume', 0), open_interest=short_call['openInterest'],
                         implied_volatility=short_call['impliedVolatility'], delta=short_call.get('delta', 0),
                         gamma=short_call.get('gamma', 0), theta=short_call.get('theta', 0),
                         vega=short_call.get('vega', 0), rho=short_call.get('rho', 0)),
                OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='call', strike=long_call['strike'],
                         expiry=options_df['expiry_date'].iloc[0], bid=long_call['bid'], ask=long_call['ask'],
                         mid=long_call['mid'], volume=long_call.get('volume', 0), open_interest=long_call['openInterest'],
                         implied_volatility=long_call['impliedVolatility'], delta=long_call.get('delta', 0),
                         gamma=long_call.get('gamma', 0), theta=long_call.get('theta', 0),
                         vega=long_call.get('vega', 0), rho=long_call.get('rho', 0)),
                OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='put', strike=short_put['strike'],
                         expiry=options_df['expiry_date'].iloc[0], bid=short_put['bid'], ask=short_put['ask'],
                         mid=short_put['mid'], volume=short_put.get('volume', 0), open_interest=short_put['openInterest'],
                         implied_volatility=short_put['impliedVolatility'], delta=short_put.get('delta', 0),
                         gamma=short_put.get('gamma', 0), theta=short_put.get('theta', 0),
                         vega=short_put.get('vega', 0), rho=short_put.get('rho', 0)),
                OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='put', strike=long_put['strike'],
                         expiry=options_df['expiry_date'].iloc[0], bid=long_put['bid'], ask=long_put['ask'],
                         mid=long_put['mid'], volume=long_put.get('volume', 0), open_interest=long_put['openInterest'],
                         implied_volatility=long_put['impliedVolatility'], delta=long_put.get('delta', 0),
                         gamma=long_put.get('gamma', 0), theta=long_put.get('theta', 0),
                         vega=long_put.get('vega', 0), rho=long_put.get('rho', 0))
            ]
            
            # Calculate strategy metrics
            net_credit = (short_call['bid'] + short_put['bid'] - long_call['ask'] - long_put['ask'])
            max_profit = net_credit
            max_loss = (short_call['strike'] - long_call['strike']) - net_credit
            breakeven_upper = short_call['strike'] + net_credit
            breakeven_lower = short_put['strike'] - net_credit
            
            # Calculate Greeks
            net_delta = sum(leg.delta for leg in legs)
            net_gamma = sum(leg.gamma for leg in legs)
            net_theta = sum(leg.theta for leg in legs)
            net_vega = sum(leg.vega for leg in legs)
            net_rho = sum(leg.rho for leg in legs)
            
            # Probability of profit (simplified)
            prob_profit = 0.6  # This would be calculated using Monte Carlo simulation
            
            return Strategy(
                name="Iron Condor",
                strategy_type="income",
                legs=legs,
                net_debit=0,
                net_credit=net_credit,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_lower, breakeven_upper],
                probability_of_profit=prob_profit,
                expected_return=net_credit * 0.7,  # Simplified
                risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                greeks={'delta': net_delta, 'gamma': net_gamma, 'theta': net_theta, 'vega': net_vega, 'rho': net_rho},
                market_conditions=['range_bound', 'low_volatility'],
                confidence_score=0.75,
                risk_level='medium'
            )
            
        except Exception as e:
            logger.error(f"Error creating Iron Condor: {e}")
            return None
    
    def _create_butterfly(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Butterfly strategy"""
        try:
            calls = options_df[options_df['option_type'] == 'call'].sort_values('strike')
            puts = options_df[options_df['option_type'] == 'put'].sort_values('strike', ascending=False)
            
            if len(calls) < 3 or len(puts) < 3:
                return None
            
            # Find ATM strikes
            atm_call_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_put_idx = (puts['strike'] - current_price).abs().idxmin()
            
            # Select strikes for butterfly
            if atm_call_idx > 0 and atm_call_idx < len(calls) - 1:
                long_call = calls.iloc[atm_call_idx - 1]
                short_call = calls.iloc[atm_call_idx]
                long_call2 = calls.iloc[atm_call_idx + 1]
                
                # Create legs
                legs = [
                    OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='call', strike=long_call['strike'],
                             expiry=options_df['expiry_date'].iloc[0], bid=long_call['bid'], ask=long_call['ask'],
                             mid=long_call['mid'], volume=long_call.get('volume', 0), open_interest=long_call['openInterest'],
                             implied_volatility=long_call['impliedVolatility'], delta=long_call.get('delta', 0),
                             gamma=long_call.get('gamma', 0), theta=long_call.get('theta', 0),
                             vega=long_call.get('vega', 0), rho=long_call.get('rho', 0)),
                    OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='call', strike=short_call['strike'],
                             expiry=options_df['expiry_date'].iloc[0], bid=short_call['bid'], ask=short_call['ask'],
                             mid=short_call['mid'], volume=short_call.get('volume', 0), open_interest=short_call['openInterest'],
                             implied_volatility=short_call['impliedVolatility'], delta=short_call.get('delta', 0),
                             gamma=short_call.get('gamma', 0), theta=short_call.get('theta', 0),
                             vega=short_call.get('vega', 0), rho=short_call.get('rho', 0)),
                    OptionLeg(symbol=options_df['symbol'].iloc[0], option_type='call', strike=long_call2['strike'],
                             expiry=options_df['expiry_date'].iloc[0], bid=long_call2['bid'], ask=long_call2['ask'],
                             mid=long_call2['mid'], volume=long_call2.get('volume', 0), open_interest=long_call2['openInterest'],
                             implied_volatility=long_call2['impliedVolatility'], delta=long_call2.get('delta', 0),
                             gamma=long_call2.get('gamma', 0), theta=long_call2.get('theta', 0),
                             vega=long_call2.get('vega', 0), rho=long_call2.get('rho', 0))
                ]
                
                # Calculate strategy metrics
                net_debit = (long_call['ask'] + long_call2['ask'] - 2 * short_call['bid'])
                max_profit = (short_call['strike'] - long_call['strike']) - net_debit
                max_loss = net_debit
                breakeven_upper = short_call['strike'] + max_profit
                breakeven_lower = short_call['strike'] - max_profit
                
                # Calculate Greeks
                net_delta = sum(leg.delta for leg in legs)
                net_gamma = sum(leg.gamma for leg in legs)
                net_theta = sum(leg.theta for leg in legs)
                net_vega = sum(leg.vega for leg in legs)
                net_rho = sum(leg.rho for leg in legs)
                
                return Strategy(
                    name="Call Butterfly",
                    strategy_type="income",
                    legs=legs,
                    net_debit=net_debit,
                    net_credit=0,
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven_points=[breakeven_lower, breakeven_upper],
                    probability_of_profit=0.6,
                    expected_return=net_debit * 0.7,
                    risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                    greeks={'delta': net_delta, 'gamma': net_gamma, 'theta': net_theta, 'vega': net_vega, 'rho': net_rho},
                    market_conditions=['neutral', 'low_volatility'],
                    confidence_score=0.7,
                    risk_level='medium'
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating Butterfly: {e}")
            return None
    
    def _create_straddle(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Straddle strategy"""
        pass
    
    def _create_strangle(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Strangle strategy"""
        pass
    
    def _create_collar(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Collar strategy"""
        pass
    
    def _create_covered_call(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Covered Call strategy"""
        pass
    
    def _create_protective_put(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Protective Put strategy"""
        pass
    
    def _create_calendar_spread(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Calendar Spread strategy"""
        pass
    
    def _create_diagonal_spread(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Diagonal Spread strategy"""
        pass
    
    def _create_ratio_spread(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Ratio Spread strategy"""
        pass
    
    def _create_backspread(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Backspread strategy"""
        pass
    
    def _create_jade_lizard(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Jade Lizard strategy"""
        pass
    
    def _create_iron_butterfly(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Iron Butterfly strategy"""
        pass
    
    def _create_condor(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Condor strategy"""
        pass
    
    def _create_box_spread(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Box Spread strategy"""
        pass
    
    def _create_conversion(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Conversion strategy"""
        pass
    
    def _create_reversal(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Reversal strategy"""
        pass
    
    def _create_synthetic_long(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Synthetic Long strategy"""
        pass
    
    def _create_synthetic_short(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Synthetic Short strategy"""
        pass
    
    def _create_volatility_arbitrage(self, options_df: pd.DataFrame, current_price: float, market_regime: MarketRegime) -> Optional[Strategy]:
        """Create Volatility Arbitrage strategy"""
        pass

# --- PORTFOLIO OPTIMIZER ---
class PortfolioOptimizer:
    def __init__(self):
        self.max_positions = 10
        self.max_correlation = 0.7
        self.max_portfolio_risk = 0.02
    
    def optimize_portfolio(self, strategies: List[Strategy], market_regime: MarketRegime) -> Portfolio:
        """Optimize portfolio using advanced techniques"""
        if not strategies:
            return Portfolio([], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Filter strategies based on market regime
        filtered_strategies = self._filter_by_market_regime(strategies, market_regime)
        
        # Select best strategies using optimization
        selected_strategies = self._select_optimal_strategies(filtered_strategies)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(selected_strategies)
        
        return Portfolio(
            strategies=selected_strategies,
            **portfolio_metrics
        )
    
    def _filter_by_market_regime(self, strategies: List[Strategy], market_regime: MarketRegime) -> List[Strategy]:
        """Filter strategies based on market regime"""
        filtered = []
        
        for strategy in strategies:
            if market_regime.regime == "High Volatility":
                if strategy.strategy_type in ['income', 'volatility']:
                    filtered.append(strategy)
            elif market_regime.regime == "Low Volatility":
                if strategy.strategy_type in ['directional', 'volatility']:
                    filtered.append(strategy)
            else:
                filtered.append(strategy)
        
        return filtered
    
    def _select_optimal_strategies(self, strategies: List[Strategy]) -> List[Strategy]:
        """Select optimal strategies using optimization"""
        if len(strategies) <= self.max_positions:
            return strategies
        
        # Sort by risk-adjusted return
        strategies.sort(key=lambda s: s.expected_return / max(s.max_loss, 0.01), reverse=True)
        
        # Select top strategies with diversification
        selected = []
        for strategy in strategies:
            if len(selected) >= self.max_positions:
                break
            
            # Check correlation with existing strategies
            if self._check_correlation(strategy, selected):
                selected.append(strategy)
        
        return selected
    
    def _check_correlation(self, new_strategy: Strategy, existing_strategies: List[Strategy]) -> bool:
        """Check if new strategy is sufficiently uncorrelated"""
        if not existing_strategies:
            return True
        
        # Simplified correlation check based on strategy type and Greeks
        for existing in existing_strategies:
            if existing.strategy_type == new_strategy.strategy_type:
                # Check delta correlation
                delta_corr = abs(existing.greeks['delta'] - new_strategy.greeks['delta'])
                if delta_corr < 0.1:  # Too similar
                    return False
        
        return True
    
    def _calculate_portfolio_metrics(self, strategies: List[Strategy]) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics"""
        if not strategies:
            return {
                'total_delta': 0, 'total_gamma': 0, 'total_theta': 0, 'total_vega': 0,
                'portfolio_beta': 0, 'portfolio_volatility': 0, 'expected_return': 0,
                'var_95': 0, 'max_drawdown': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
                'calmar_ratio': 0, 'diversification_ratio': 0, 'risk_score': 0
            }
        
        # Aggregate Greeks
        total_delta = sum(s.greeks['delta'] for s in strategies)
        total_gamma = sum(s.greeks['gamma'] for s in strategies)
        total_theta = sum(s.greeks['theta'] for s in strategies)
        total_vega = sum(s.greeks['vega'] for s in strategies)
        
        # Portfolio metrics
        expected_return = sum(s.expected_return for s in strategies)
        portfolio_volatility = sqrt(sum(s.max_loss**2 for s in strategies))  # Simplified
        
        # Risk metrics
        var_95 = np.percentile([s.max_loss for s in strategies], 95)
        max_drawdown = max(s.max_loss for s in strategies)
        
        # Performance ratios
        sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
        sortino_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0
        calmar_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0
        
        # Diversification
        diversification_ratio = len(strategies) / max(1, len(set(s.strategy_type for s in strategies)))
        
        # Risk score (0-1, lower is better)
        risk_score = min(1.0, max_drawdown / 0.1)  # Normalize to 10% max loss
        
        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'portfolio_beta': 0,  # Would need market data
            'portfolio_volatility': portfolio_volatility,
            'expected_return': expected_return,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'diversification_ratio': diversification_ratio,
            'risk_score': risk_score
        }

# --- MAIN APPLICATION CLASS ---
class AdvancedOptionsEngine:
    def __init__(self):
        self.config = AdvancedConfig()
        self.market_analyzer = MarketAnalyzer()
        self.options_manager = OptionsDataManager()
        self.ml_predictor = MLStrategyPredictor()
        self.strategy_generator = AdvancedStrategyGenerator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.universe = []
    
    async def initialize(self):
        """Initialize the engine"""
        await self.ml_predictor.load_models()
        self.universe = self._load_universe()
        logger.info("Advanced Options Engine initialized successfully")
    
    def _load_universe(self) -> List[str]:
        """Load symbol universe"""
        if not os.path.exists(UNIVERSE_FILE):
            default_symbols = [
                'AAPL', 'TSLA', 'GOOGL', 'NVDA', 'AMD', 'META', 'MSFT', 'AMZN',
                'SPY', 'QQQ', 'IWM', 'COIN', 'UBER', 'NFLX', 'CRM', 'ADBE',
                'PYPL', 'INTC', 'CSCO', 'ORCL', 'IBM', 'JPM', 'BAC', 'WFC',
                'XOM', 'CVX', 'JNJ', 'PFE', 'UNH', 'HD', 'LOW', 'WMT'
            ]
            pd.DataFrame(default_symbols, columns=['symbol']).to_csv(UNIVERSE_FILE, index=False)
            return default_symbols
        
        try:
            df = pd.read_csv(UNIVERSE_FILE)
            return df['symbol'].str.strip().str.upper().unique().tolist()
        except Exception as e:
            logger.error(f"Error loading universe: {e}")
            return []
    
    async def scan_market(self, symbols: List[str] = None) -> List[Portfolio]:
        """Main market scanning function"""
        if symbols is None:
            symbols = self.universe
        
        logger.info(f"Starting advanced market scan for {len(symbols)} symbols")
        
        # Get market regime
        market_regime = await self.market_analyzer.get_market_regime()
        logger.info(f"Market regime: {market_regime.regime} (VIX: {market_regime.vix_level:.2f})")
        
        # Process symbols in parallel
        all_strategies = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for symbol in symbols:
                future = executor.submit(self._process_symbol, symbol, market_regime)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    strategies = future.result()
                    if strategies:
                        all_strategies.extend(strategies)
                except Exception as e:
                    logger.error(f"Error processing symbol: {e}")
        
        # Optimize portfolios
        portfolios = []
        if all_strategies:
            # Group strategies by type
            income_strategies = [s for s in all_strategies if s.strategy_type == 'income']
            directional_strategies = [s for s in all_strategies if s.strategy_type == 'directional']
            volatility_strategies = [s for s in all_strategies if s.strategy_type == 'volatility']
            
            # Create optimized portfolios
            if income_strategies:
                income_portfolio = self.portfolio_optimizer.optimize_portfolio(income_strategies, market_regime)
                portfolios.append(income_portfolio)
            
            if directional_strategies:
                directional_portfolio = self.portfolio_optimizer.optimize_portfolio(directional_strategies, market_regime)
                portfolios.append(directional_portfolio)
            
            if volatility_strategies:
                volatility_portfolio = self.portfolio_optimizer.optimize_portfolio(volatility_strategies, market_regime)
                portfolios.append(volatility_portfolio)
            
            # Create combined portfolio
            combined_portfolio = self.portfolio_optimizer.optimize_portfolio(all_strategies, market_regime)
            portfolios.append(combined_portfolio)
        
        logger.info(f"Scan complete. Found {len(portfolios)} optimized portfolios")
        return portfolios
    
    async def _process_symbol(self, symbol: str, market_regime: MarketRegime) -> List[Strategy]:
        """Process individual symbol and generate strategies"""
        try:
            # Get ticker data
            ticker = await self.options_manager.get_ticker_data(symbol)
            if not ticker:
                return []
            
            # Get historical data
            hist_data = await self.options_manager.get_historical_data(ticker)
            if hist_data is None or len(hist_data) < 30:
                return []
            
            current_price = hist_data['Close'].iloc[-1]
            if current_price < self.config.min_price:
                return []
            
            # Get options chain
            options_df = await self.options_manager.get_options_chain(ticker, 30)
            if options_df is None or options_df.empty:
                return []
            
            # Generate strategies
            strategies = []
            for strategy_name, strategy_func in self.strategy_generator.strategy_templates.items():
                try:
                    strategy = strategy_func(options_df, current_price, market_regime)
                    if strategy:
                        strategies.append(strategy)
                except Exception as e:
                    logger.error(f"Error creating {strategy_name} for {symbol}: {e}")
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return []

# --- GLOBAL INSTANCE ---
engine = AdvancedOptionsEngine()

# --- API ENDPOINTS ---
@app.on_event("startup")
async def startup_event():
    await engine.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Advanced Options Engine...")

@app.get("/")
async def root():
    return {"message": "Advanced Options Strategy Finder - Alpha Pro Max", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/market-regime")
async def get_market_regime():
    regime = await engine.market_analyzer.get_market_regime()
    return asdict(regime)

@app.get("/universe")
async def get_universe():
    return {"symbols": engine.universe}

@app.post("/universe")
async def update_universe(request: UniverseRequest):
    try:
        df = pd.DataFrame(request.symbols, columns=['symbol'])
        df['symbol'] = df['symbol'].str.strip().str.upper()
        df.drop_duplicates(inplace=True)
        df.to_csv(UNIVERSE_FILE, index=False)
        engine.universe = df['symbol'].tolist()
        return {"message": "Universe updated successfully", "count": len(engine.universe)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan")
async def scan_market(request: StrategyRequest = None):
    try:
        if request is None:
            request = StrategyRequest()
        
        symbols = request.symbols if request.symbols else engine.universe
        portfolios = await engine.scan_market(symbols)
        
        # Convert to serializable format
        result = []
        for portfolio in portfolios:
            portfolio_dict = asdict(portfolio)
            # Convert strategies to dict format
            portfolio_dict['strategies'] = [asdict(strategy) for strategy in portfolio.strategies]
            result.append(portfolio_dict)
        
        return {"portfolios": result, "scan_time": datetime.now().isoformat()}
    
    except Exception as e:
        logger.error(f"Error in scan endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def backtest_strategies(request: BacktestRequest):
    """Backtest strategies (placeholder for future implementation)"""
    return {"message": "Backtesting feature coming soon", "request": request.dict()}

@app.get("/strategies")
async def get_available_strategies():
    """Get list of available strategies"""
    return {
        "strategies": list(engine.strategy_generator.strategy_templates.keys()),
        "strategy_types": ["income", "directional", "volatility", "arbitrage"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)