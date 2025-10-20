# V15 - Market Regime Awareness, Smarter Spreads, and Volatility Term Structure
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import time
import yfinance as yf
import os

# --- CONFIGURATION & SETUP ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UNIVERSE_FILE = "universe.csv"
yf_cache = {}

# --- DATACLASSES FOR TYPE HINTING & API MODELS ---
@dataclass
class Config:
    risk_free_rate: float = 0.045
    min_price: float = 10.0
    min_oi: int = 100
    max_bid_ask_spread_pct: float = 0.50

class UniverseRequest(BaseModel):
    symbols: List[str]

# --- UNIVERSE & MARKET REGIME ---
def get_universe() -> List[str]:
    # ... (implementation from V14)
    if not os.path.exists(UNIVERSE_FILE):
        default_symbols = ['AAPL', 'TSLA', 'GOOG', 'NVDA', 'AMD', 'META', 'SPY', 'QQQ', 'IWM', 'COIN', 'MSFT', 'AMZN']
        pd.DataFrame(default_symbols, columns=['symbol']).to_csv(UNIVERSE_FILE, index=False)
        return default_symbols
    try:
        df = pd.read_csv(UNIVERSE_FILE)
        return df['symbol'].str.strip().str.upper().unique().tolist()
    except Exception as e:
        logger.error(f"Error reading universe file: {e}")
        return []

def get_market_regime() -> Tuple[str, float]:
    try:
        vix = yf.Ticker("^VIX")
        vix_price = vix.history(period="1d")['Close'].iloc[-1]
        if vix_price > 25:
            regime = "High Volatility"
        elif vix_price < 15:
            regime = "Low Volatility"
        else:
            regime = "Neutral"
        return regime, vix_price
    except Exception as e:
        logger.error(f"Could not fetch VIX data: {e}")
        return "Unknown", 0.0

# --- CORE CALCULATION FUNCTIONS (ROBUST YFINANCE MODEL) ---
def get_ticker_yf(symbol: str) -> Optional[yf.Ticker]:
    # ... (implementation from V14)
    if symbol in yf_cache and (time.time() - yf_cache[symbol]['timestamp'] < 300):
        return yf_cache[symbol]['ticker']
    try:
        ticker = yf.Ticker(symbol)
        if ticker.info.get('regularMarketPrice') is not None:
            yf_cache[symbol] = {'ticker': ticker, 'timestamp': time.time()}
            return ticker
        else:
            return None
    except Exception:
        return None

def get_historical_data_yf(ticker: yf.Ticker) -> Optional[pd.DataFrame]:
    # ... (implementation from V14)
    try:
        history = ticker.history(period="1y")
        if history.empty: return None
        history.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        return history
    except Exception: return None

def get_options_chain_yf(ticker: yf.Ticker, config: Config, days_from_now: int) -> Optional[pd.DataFrame]:
    try:
        expirations = ticker.options
        today = datetime.now().date()
        valid_expiries = [(e, (datetime.strptime(e, '%Y-%m-%d').date() - today).days) for e in expirations if (datetime.strptime(e, '%Y-%m-%d').date() - today).days > 5]
        if not valid_expiries: return None
        target_expiry, _ = min(valid_expiries, key=lambda x: abs(x[1] - days_from_now))
        chain = ticker.option_chain(target_expiry)
        df = pd.concat([chain.calls.assign(option_type='call'), chain.puts.assign(option_type='put')])
        if df.empty: return None
        df.rename(columns={'openInterest': 'openInterest', 'impliedVolatility': 'impliedVolatility', 'strike': 'strike', 'bid':'bid', 'ask':'ask'}, inplace=True)
        df = df[(df['openInterest'] >= config.min_oi) & (df['bid'] > 0) & (df['ask'] > 0)].copy()
        df['spread_pct'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
        df = df[df['spread_pct'] <= config.max_bid_ask_spread_pct]
        df['expiry_date'] = target_expiry
        return df if not df.empty else None
    except Exception: return None

def calculate_hv20(history: pd.DataFrame) -> float: # ... (implementation from V14)
    if len(history) < 21: return np.nan
    log_returns = np.log(history['Close'] / history['Close'].shift(1))
    return log_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

def calculate_parkinson20(history: pd.DataFrame) -> float: # ... (implementation from V14)
    if len(history) < 20: return np.nan
    ln_hl_squared = (np.log(history['High'] / history['Low']) ** 2)
    x = ln_hl_squared.rolling(20).mean().iloc[-1]
    if pd.isna(x) or x <= 0: return np.nan
    return np.sqrt(x / (4 * np.log(2))) * np.sqrt(252) * 100

def calculate_variance_ratio(history: pd.DataFrame, n: int = 20) -> Tuple[float, str]: # ... (implementation from V14)
    if len(history) < n + 1: return np.nan, "neutral"
    log_returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()
    if len(log_returns) < n: return np.nan, "neutral"
    var1 = log_returns.var()
    varN = log_returns.rolling(n).sum().var()
    if var1 == 0: return np.nan, "neutral"
    vr = varN / (n * var1)
    signal = "trend" if vr > 1.05 else "meanrev" if vr < 0.95 else "neutral"
    return vr, signal

def get_atm_iv(options_df: pd.DataFrame, spot: float) -> float: # ... (implementation from V14)
    if options_df is None or options_df.empty: return np.nan
    atm_range = 2.0 if spot < 50 else spot * 0.05
    atm_options = options_df[abs(options_df['strike'] - spot) <= atm_range]
    if len(atm_options) < 2:
        atm_options = options_df.iloc[(options_df['strike'] - spot).abs().argsort()[:5]]
    if atm_options.empty: return np.nan
    valid_ivs = atm_options['impliedVolatility'][(atm_options['impliedVolatility'] > 0.01) & (atm_options['impliedVolatility'] < 10.0)]
    return valid_ivs.median() if not valid_ivs.empty else np.nan

# --- SMARTER STRATEGY & RISK LOGIC ---

def score_strategies(data: dict) -> Tuple[str, str, List[str]]:
    scores = {"TREND_FOLLOW": 0, "MEAN_REVERSION": 0}
    signals = []
    
    # Primary Signals
    if data.get('vr_signal') == "trend": scores["TREND_FOLLOW"] += 1.2; signals.append(f"✅ Strong Trend Signal (VR={data['vr_20']:.2f})")
    if data.get('vr_signal') == "meanrev": scores["MEAN_REVERSION"] += 1.2; signals.append(f"✅ Mean Reversion Signal (VR={data['vr_20']:.2f})")
    if data.get('iv_hv_gap', 0) > 5.0: scores["MEAN_REVERSION"] += 1.5; signals.append(f"✅ High IV Premium (Gap: {data['iv_hv_gap']:.1f} pts)")
    
    # Contextual Boosts from Market Regime
    regime = data.get('market_regime', 'Neutral')
    if regime == "High Volatility":
        scores["MEAN_REVERSION"] *= 1.5 # Boost mean reversion in high vol
    elif regime == "Low Volatility":
        scores["TREND_FOLLOW"] *= 1.5 # Boost trend following in low vol

    # Term Structure Signal
    if data.get('term_structure_slope', 0) < -1.0: # Backwardation
        scores["MEAN_REVERSION"] += 2.0 # Strong signal to sell front-month vol
        signals.append(f"🔥 Backwardation Detected (Slope: {data['term_structure_slope']:.2f})")

    best_strategy = max(scores, key=scores.get)
    if scores[best_strategy] < 1.0: return "NEUTRAL_WAIT", "No strong signal confluence.", ["- No conclusive signals"]
    
    rationales = {"TREND_FOLLOW": "Momentum signals suggest the current trend will continue.", "MEAN_REVERSION": "Volatility is overpriced and/or the stock is range-bound."}
    return best_strategy, rationales.get(best_strategy, "Multiple signals."), signals

def generate_risk_and_trade_idea(idea: dict, options_df: pd.DataFrame, history: pd.DataFrame) -> dict:
    strategy, price, symbol, expiry = idea['strategy'], idea['current_price'], idea['symbol'], idea['expiry_date']
    calls = options_df[options_df['option_type'] == 'call'].sort_values('strike').reset_index(drop=True)
    puts = options_df[options_df['option_type'] == 'put'].sort_values('strike', ascending=False).reset_index(drop=True)
    if calls.empty or puts.empty: return {"trade": ["Could not find liquid options."], "exit": "N/A"}

    atm_call_idx = (calls['strike'] - price).abs().idxmin()
    atm_put_idx = (puts['strike'] - price).abs().idxmin()
    
    trade_legs, exit_strategy = [], ""
    max_loss, invalidation_point, hedge_idea = "N/A", "N/A", "N/A"
    def format_leg(leg, action): return f"{action} 1x {symbol} {expiry} {leg['strike']} {leg['option_type'].upper()} (@ {leg['bid']:.2f} - {leg['ask']:.2f})"

    # Smarter Strategy Construction
    if strategy == "TREND_FOLLOW": # Construct a Debit Spread
        leg1 = calls.iloc[atm_call_idx]
        leg2 = calls.iloc[atm_call_idx + 2] if atm_call_idx + 2 < len(calls) else calls.iloc[-1]
        trade_legs.extend([format_leg(leg1, "BUY"), format_leg(leg2, "SELL")])
        net_debit = (leg1['ask'] - leg2['bid']) * 100
        trade_legs.append(f"NET DEBIT: ${net_debit:.2f} (LIMIT)")
        max_loss = f"${net_debit:.2f}"
        invalidation_point = f"A close below the 20-day low of ${history['Low'].tail(20).min():.2f}."
        exit_strategy = "Exit for profit when gain is 50% of max profit. Max loss is the debit paid."
        hedge_idea = "This is a risk-defined strategy and does not require an additional hedge."

    elif strategy == "MEAN_REVERSION": # Construct an Iron Condor
        short_put = puts.iloc[atm_put_idx + 2] if atm_put_idx + 2 < len(puts) else puts.iloc[-1]
        long_put = puts.iloc[atm_put_idx + 4] if atm_put_idx + 4 < len(puts) else puts.iloc[-1]
        short_call = calls.iloc[atm_call_idx + 2] if atm_call_idx + 2 < len(calls) else calls.iloc[-1]
        long_call = calls.iloc[atm_call_idx + 4] if atm_call_idx + 4 < len(calls) else calls.iloc[-1]
        trade_legs.extend([format_leg(short_put, "SELL"), format_leg(long_put, "BUY"), format_leg(short_call, "SELL"), format_leg(long_call, "BUY")])
        net_credit = (short_put['bid'] - long_put['ask'] + short_call['bid'] - long_call['ask']) * 100
        trade_legs.append(f"NET CREDIT: ${net_credit:.2f} (LIMIT)")
        max_loss = f"${((short_put['strike'] - long_put['strike']) * 100) - net_credit:.2f}"
        invalidation_point = f"A close outside the short strikes ({short_put['strike']} - {short_call['strike']})."
        exit_strategy = "Close for profit when you capture 50% of the initial credit. Manage early if price challenges a short strike."
        hedge_idea = "This is a risk-defined strategy and does not require an additional hedge."
    
    else: return {"trade": ["No clear signals."], "exit": "N/A", "max_loss": "N/A", "invalidation_point": "N/A", "hedge_idea": "N/A"}
    
    return {"trade": trade_legs, "exit": exit_strategy, "max_loss": max_loss, "invalidation_point": invalidation_point, "hedge_idea": hedge_idea}


def pass_one_process_symbol(symbol: str, config: Config) -> Optional[dict]:
    ticker = get_ticker_yf(symbol)
    if ticker is None: return None

    history = get_historical_data_yf(ticker)
    if history is None or len(history) < 30: return None

    current_price = history['Close'].iloc[-1]
    if current_price < config.min_price: return None

    options_df_30 = get_options_chain_yf(ticker, config, 30)
    options_df_90 = get_options_chain_yf(ticker, config, 90)
    if options_df_30 is None or options_df_90 is None: return None
    
    iv30 = get_atm_iv(options_df_30, current_price)
    iv90 = get_atm_iv(options_df_90, current_price)
    hv20 = calculate_hv20(history)
    parkinson20 = calculate_parkinson20(history)
    vr_20, vr_signal = calculate_variance_ratio(history)
    if any(map(pd.isna, [iv30, iv90, hv20, parkinson20, vr_20])): return None
        
    return {
        "symbol": symbol, "current_price": current_price, "hv20": hv20, "parkinson20": parkinson20,
        "vr_20": vr_20, "vr_signal": vr_signal, "p_vs_hv_flag": 1 if parkinson20 > 1.67 * hv20 else 0,
        "iv30": iv30 * 100, "iv90": iv90 * 100,
        "term_structure_slope": (iv90 - iv30) * 100,
        "iv_hv_gap": (iv30 * 100) - hv20, 
        "options_df_json": options_df_30.to_json(orient='split'),
        "history_json": history.to_json(orient='split'),
        "expiry_date": options_df_30['expiry_date'].iloc[0]
    }

def pass_two_enrich_data(ideas: List[dict], market_regime: str) -> List[dict]:
    if not ideas: return []
    df = pd.DataFrame(ideas)
    # df['skew_pct'] is no longer used but can be added back if needed
    df['asym_score'] = df['iv_hv_gap'] + df['term_structure_slope']
    enriched_ideas = []
    for _, row in df.iterrows():
        idea_dict = row.to_dict()
        idea_dict['market_regime'] = market_regime # Pass regime to scoring
        strategy, rationale, confluence = score_strategies(idea_dict)
        idea_dict.update({'strategy': strategy, 'rationale': rationale, 'confluence': confluence})
        
        options_df = pd.read_json(idea_dict['options_df_json'], orient='split')
        history_df = pd.read_json(idea_dict['history_json'], orient='split')
        idea_dict.update(generate_risk_and_trade_idea(idea_dict, options_df, history_df))
        
        del idea_dict['options_df_json']
        del idea_dict['history_json']
        enriched_ideas.append(idea_dict)
    return sorted(enriched_ideas, key=lambda x: x.get('asym_score', 0), reverse=True)

# --- API ENDPOINTS ---
@app.get("/get-market-regime")
async def get_market_regime_endpoint():
    regime, vix = get_market_regime()
    return {"regime": regime, "vix_level": vix}

@app.get("/get-universe")
async def get_universe_endpoint():
    return {"symbols": get_universe()}

@app.post("/update-universe")
async def update_universe_endpoint(request: UniverseRequest):
    try:
        df = pd.DataFrame(request.symbols, columns=['symbol'])
        df['symbol'] = df['symbol'].str.strip().str.upper()
        df.drop_duplicates(inplace=True)
        df.to_csv(UNIVERSE_FILE, index=False)
        return {"message": "Universe updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update universe file: {e}")

@app.post("/scan")
async def run_scan():
    logger.info(f"Received scan request.")
    config = Config()
    regime, _ = get_market_regime()
    symbols_to_scan = get_universe()
    pass_one_results = []
    for s in symbols_to_scan:
        logger.info(f"--- Processing {s} ---")
        res = pass_one_process_symbol(s, config)
        if res is not None:
            pass_one_results.append(res)
        time.sleep(1) 

    if not pass_one_results:
        return {"error": "No valid data could be retrieved for the provided symbols."}
    return pass_two_enrich_data(pass_one_results, regime)

