# V13 - Removed unused alpaca_trade_api import to prevent dependency conflicts.
from fastapi import FastAPI
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

# In-memory cache for yfinance Ticker objects
yf_cache = {}

# --- DATACLASSES FOR TYPE HINTING & API MODELS ---
@dataclass
class Config:
    risk_free_rate: float = 0.045
    min_price: float = 10.0
    min_oi: int = 100
    max_bid_ask_spread_pct: float = 0.50

class ScanRequest(BaseModel):
    symbols: List[str]

# --- CORE CALCULATION FUNCTIONS (ROBUST YFINANCE MODEL) ---

def get_ticker_yf(symbol: str) -> Optional[yf.Ticker]:
    """Gets a yfinance Ticker object with caching to avoid redundant calls."""
    if symbol in yf_cache and (time.time() - yf_cache[symbol]['timestamp'] < 300): # 5 min cache
        return yf_cache[symbol]['ticker']
    try:
        ticker = yf.Ticker(symbol)
        # A quick check to ensure the ticker is valid
        if ticker.info.get('regularMarketPrice') is not None:
            yf_cache[symbol] = {'ticker': ticker, 'timestamp': time.time()}
            return ticker
        else:
            logger.warning(f"Could not get valid info for {symbol} from yfinance.")
            return None
    except Exception as e:
        logger.error(f"Error creating yfinance Ticker for {symbol}: {e}")
        return None

def get_historical_data_yf(ticker: yf.Ticker) -> Optional[pd.DataFrame]:
    try:
        history = ticker.history(period="1y")
        if history.empty: return None
        # Standardize column names
        history.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'}, inplace=True)
        return history
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker.ticker} from yfinance: {e}")
        return None

def get_options_chain_yf(ticker: yf.Ticker, config: Config) -> Optional[pd.DataFrame]:
    try:
        # Find expiry date closest to 30 days out
        expirations = ticker.options
        today = datetime.now().date()
        valid_expiries = [(e, (datetime.strptime(e, '%Y-%m-%d').date() - today).days) for e in expirations if (datetime.strptime(e, '%Y-%m-%d').date() - today).days > 0]
        if not valid_expiries: return None
        target_expiry, _ = min(valid_expiries, key=lambda x: abs(x[1] - 30))

        # Fetch the options chain for that expiry
        chain = ticker.option_chain(target_expiry)
        calls = chain.calls.assign(option_type='call')
        puts = chain.puts.assign(option_type='put')
        df = pd.concat([calls, puts])

        if df.empty: return None

        # Standardize and filter
        df.rename(columns={'openInterest': 'openInterest', 'impliedVolatility': 'impliedVolatility', 'strike': 'strike', 'bid':'bid', 'ask':'ask'}, inplace=True)

        # Liquidity filter
        df = df[
            (df['openInterest'] >= config.min_oi) &
            (df['bid'] > 0) & (df['ask'] > 0)
        ].copy()
        df['spread_pct'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
        df = df[df['spread_pct'] <= config.max_bid_ask_spread_pct]
        df['expiry_date'] = target_expiry
        
        return df if not df.empty else None

    except Exception as e:
        logger.error(f"Error fetching options chain for {ticker.ticker} from yfinance: {e}")
        return None


# --- ALL OTHER CALCULATION & LOGIC FUNCTIONS REMAIN THE SAME ---
def calculate_hv20(history: pd.DataFrame) -> float:
    if len(history) < 21: return np.nan
    log_returns = np.log(history['Close'] / history['Close'].shift(1))
    return log_returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

def calculate_parkinson20(history: pd.DataFrame) -> float:
    if len(history) < 20: return np.nan
    ln_hl_squared = (np.log(history['High'] / history['Low']) ** 2)
    x = ln_hl_squared.rolling(20).mean().iloc[-1]
    if pd.isna(x) or x <= 0: return np.nan
    return np.sqrt(x / (4 * np.log(2))) * np.sqrt(252) * 100

def calculate_variance_ratio(history: pd.DataFrame, n: int = 20) -> Tuple[float, str]:
    if len(history) < n + 1: return np.nan, "neutral"
    log_returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()
    if len(log_returns) < n: return np.nan, "neutral"
    var1 = log_returns.var()
    varN = log_returns.rolling(n).sum().var()
    if var1 == 0: return np.nan, "neutral"
    vr = varN / (n * var1)
    signal = "trend" if vr > 1.05 else "meanrev" if vr < 0.95 else "neutral"
    return vr, signal

def get_atm_iv(options_df: pd.DataFrame, spot: float) -> float:
    if options_df is None or options_df.empty: return np.nan
    atm_range = 2.0 if spot < 50 else spot * 0.05
    atm_options = options_df[abs(options_df['strike'] - spot) <= atm_range]
    if len(atm_options) < 2:
        atm_options = options_df.iloc[(options_df['strike'] - spot).abs().argsort()[:5]]
    if atm_options.empty: return np.nan
    valid_ivs = atm_options['impliedVolatility'][(atm_options['impliedVolatility'] > 0.01) & (atm_options['impliedVolatility'] < 10.0)]
    return valid_ivs.median() if not valid_ivs.empty else np.nan
    
def calculate_skew_ratio(options_df: pd.DataFrame, spot: float) -> float:
    if options_df is None or options_df.empty: return 1.0
    calls = options_df[options_df['option_type'] == 'call']
    puts = options_df[options_df['option_type'] == 'put']
    if calls.empty or puts.empty: return 1.0
    put_band = puts[(puts['strike'] >= spot * 0.90) & (puts['strike'] <= spot * 0.95)]
    call_band = calls[(calls['strike'] >= spot * 1.05) & (calls['strike'] <= spot * 1.10)]
    put_iv = put_band['impliedVolatility'].median()
    call_iv = call_band['impliedVolatility'].median()
    if pd.isna(put_iv) or pd.isna(call_iv) or call_iv == 0: return 1.0
    return put_iv / call_iv

def score_strategies(data: dict) -> Tuple[str, str, List[str]]:
    scores = {"BARRIER_STRANGLE": 0, "RISK_REVERSAL": 0, "TREND_FOLLOW": 0, "MEAN_REVERSION": 0, "VOL_PREMIUM_SELL": 0}
    signals = []
    if data.get('skew_pct', 50) >= 95: scores["RISK_REVERSAL"] += 1.5; signals.append(f"✅ Extremely High Skew ({data['skew_pct']:.1f}th %ile)")
    if data.get('skew_pct', 50) <= 5: scores["RISK_REVERSAL"] += 1.5; signals.append(f"✅ Extremely Low Skew ({data['skew_pct']:.1f}th %ile)")
    if data.get('vr_signal') == "trend": scores["TREND_FOLLOW"] += 1.2; signals.append(f"✅ Strong Trend Signal (VR={data['vr_20']:.2f})")
    if data.get('vr_signal') == "meanrev": scores["MEAN_REVERSION"] += 1.2; signals.append(f"✅ Mean Reversion Signal (VR={data['vr_20']:.2f})")
    if data.get('p_vs_hv_flag') == 1: scores["MEAN_REVERSION"] += 0.8; signals.append(f"✅ Parkinson Vol > HV")
    if data.get('iv_hv_gap', 0) > 5.0: scores["VOL_PREMIUM_SELL"] += 1 + (data['iv_hv_gap'] / 10); signals.append(f"✅ High IV Premium (Gap: {data['iv_hv_gap']:.1f} pts)")
    best_strategy = max(scores, key=scores.get)
    if scores[best_strategy] < 1.0: return "NEUTRAL_WAIT", "No strong signal confluence.", ["- No conclusive signals"]
    rationales = { "RISK_REVERSAL": "Extreme skew suggests options are heavily pricing a move.", "TREND_FOLLOW": "Variance Ratio indicates a strong, persistent trend.", "MEAN_REVERSION": "Variance Ratio suggests the stock is range-bound.", "VOL_PREMIUM_SELL": "Implied vol is significantly higher than historical, suggesting overpriced options."}
    return best_strategy, rationales.get(best_strategy, "Multiple signals."), signals

def generate_executable_idea(idea: dict, options_df: pd.DataFrame) -> dict:
    strategy, price, symbol, expiry = idea['strategy'], idea['current_price'], idea['symbol'], idea['expiry_date']
    calls = options_df[options_df['option_type'] == 'call'].sort_values('strike').reset_index(drop=True)
    puts = options_df[options_df['option_type'] == 'put'].sort_values('strike', ascending=False).reset_index(drop=True)
    if calls.empty or puts.empty: return {"trade": ["Could not find liquid options."], "exit": "N/A"}
    
    atm_call_idx = (calls['strike'] - price).abs().idxmin()
    atm_put_idx = (puts['strike'] - price).abs().idxmin()
    
    trade_legs, net_cost, exit_strategy = [], 0.0, ""
    def format_leg(leg, action): return f"{action} 1x {symbol} {expiry} {leg['strike']} {leg['option_type'].upper()} (@ {leg['bid']:.2f} - {leg['ask']:.2f})"

    if strategy in ["TREND_FOLLOW"]:
        leg1, leg2 = calls.iloc[atm_call_idx], puts.iloc[atm_put_idx]
        trade_legs.extend([format_leg(leg1, "BUY"), format_leg(leg2, "BUY")])
        net_cost = leg1['ask'] + leg2['ask']
        trade_legs.append(f"NET DEBIT: ${net_cost * 100:.2f} (LIMIT)")
        exit_strategy = "Exit for profit at +50% of debit paid. Exit for loss at -50%."
    elif strategy in ["VOL_PREMIUM_SELL", "MEAN_REVERSION"]:
        leg1 = calls.iloc[atm_call_idx + 4] if atm_call_idx + 4 < len(calls) else calls.iloc[-1]
        leg2 = puts.iloc[atm_put_idx + 4] if atm_put_idx + 4 < len(puts) else puts.iloc[-1]
        trade_legs.extend([format_leg(leg1, "SELL"), format_leg(leg2, "SELL")])
        net_cost = leg1['bid'] + leg2['bid']
        trade_legs.append(f"NET CREDIT: ${net_cost * 100:.2f} (LIMIT)")
        exit_strategy = "Close for profit when you capture 50% of the initial credit."
    else: return {"trade": ["No clear signals for a trade."], "exit": "Wait for a better setup."}
    
    return {"trade": trade_legs, "exit": exit_strategy}


def pass_one_process_symbol(symbol: str, config: Config) -> Optional[dict]:
    ticker = get_ticker_yf(symbol)
    if ticker is None: return None

    history = get_historical_data_yf(ticker)
    if history is None or len(history) < 30:
        logger.warning(f"Could not get sufficient historical data for {symbol}")
        return None

    current_price = history['Close'].iloc[-1]
    if current_price < config.min_price:
        return None

    options_df = get_options_chain_yf(ticker, config)
    if options_df is None:
        logger.warning(f"Could not get a valid options chain for {symbol}")
        return None
    
    hv20 = calculate_hv20(history)
    parkinson20 = calculate_parkinson20(history)
    vr_20, vr_signal = calculate_variance_ratio(history)
    iv30 = get_atm_iv(options_df, current_price)

    if any(map(pd.isna, [hv20, parkinson20, vr_20, iv30])): return None
        
    return {
        "symbol": symbol, "current_price": current_price, "hv20": hv20, "parkinson20": parkinson20,
        "vr_20": vr_20, "vr_signal": vr_signal, "p_vs_hv_flag": 1 if parkinson20 > 1.67 * hv20 else 0,
        "iv30": iv30 * 100, "skew_ratio": calculate_skew_ratio(options_df, current_price),
        "iv_hv_gap": (iv30 * 100) - hv20, "options_df_json": options_df.to_json(orient='split'),
        "expiry_date": options_df['expiry_date'].iloc[0]
    }

def pass_two_enrich_data(ideas: List[dict]) -> List[dict]:
    if not ideas: return []
    df = pd.DataFrame(ideas)
    df['skew_pct'] = df['skew_ratio'].rank(pct=True) * 100
    df['asym_score'] = (df['skew_pct'] - 50).abs() + df['iv_hv_gap']
    enriched_ideas = []
    for _, row in df.iterrows():
        idea_dict = row.to_dict()
        strategy, rationale, confluence = score_strategies(idea_dict)
        idea_dict.update({'strategy': strategy, 'rationale': rationale, 'confluence': confluence})
        options_df = pd.read_json(idea_dict['options_df_json'], orient='split')
        idea_dict.update(generate_executable_idea(idea_dict, options_df))
        del idea_dict['options_df_json']
        enriched_ideas.append(idea_dict)
    return sorted(enriched_ideas, key=lambda x: x.get('asym_score', 0), reverse=True)

@app.post("/scan")
async def run_scan(request: ScanRequest):
    logger.info(f"Received scan request for symbols: {request.symbols}")
    config = Config()
    pass_one_results = []
    for s in request.symbols:
        logger.info(f"--- Processing {s} ---")
        res = pass_one_process_symbol(s, config)
        if res is not None:
            pass_one_results.append(res)
        
        # A short delay to be courteous to the unofficial yfinance API
        time.sleep(1) 

    if not pass_one_results:
        return {"error": "No valid data could be retrieved for the provided symbols."}
    return pass_two_enrich_data(pass_one_results)

