# V37 (Definitive) - Implemented flexible strike selection for trade construction.
# Loosened bid-ask spread filter to adapt to real market conditions.
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
from math import log, sqrt, exp, erf

# --- CONFIGURATION & SETUP ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [CORE] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UNIVERSE_FILE = "universe.csv"
yf_cache = {}

# --- DATACLASSES & MODELS ---
@dataclass
class Config:
    risk_free_rate: float = 0.045
    min_price: float = 10.0
    min_oi: int = 10
    # DEFINITIVE FIX: Loosened the bid-ask spread filter to be more realistic.
    max_bid_ask_spread_pct: float = 0.75

class UniverseRequest(BaseModel):
    symbols: List[str]

# --- BLACK-SCHOLES ENGINE ---
def _n_cdf(x): return (1.0 + erf(x / sqrt(2.0))) / 2.0
def _n_pdf(x): return exp(-x**2 / 2.0) / sqrt(2.0 * np.pi)

def calculate_greeks(option_type: str, S: float, K: float, T: float, r: float, iv: float) -> Tuple[float, float]:
    if T <= 0 or iv <= 0: return 0, 0
    d1 = (log(S / K) + (r + 0.5 * iv**2) * T) / (iv * sqrt(T))
    d2 = d1 - iv * sqrt(T)
    option_type = option_type[0].lower()
    if option_type == 'c':
        delta = _n_cdf(d1)
        theta = (- (S * _n_pdf(d1) * iv) / (2 * sqrt(T)) - r * K * exp(-r * T) * _n_cdf(d2)) / 365
        return delta, theta
    else: # Put
        delta = -_n_cdf(-d1)
        theta = (- (S * _n_pdf(d1) * iv) / (2 * sqrt(T)) + r * K * exp(-r * T) * _n_cdf(-d2)) / 365
        return delta, theta

# --- MARKET & UNIVERSE ANALYSIS ---
def get_universe() -> List[str]:
    if not os.path.exists(UNIVERSE_FILE):
        default_symbols = ['AAPL', 'TSLA', 'GOOG', 'NVDA', 'AMD', 'META', 'SPY', 'QQQ', 'IWM', 'COIN', 'MSFT', 'AMZN', 'UBER']
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
        vix_hist = yf.Ticker("^VIX").history(period="5d")
        if vix_hist.empty: return "Neutral", 20.0
        vix = vix_hist['Close'].iloc[-1]
        if vix > 25: return "High Volatility", vix
        elif vix < 15: return "Low Volatility", vix
        else: return "Neutral", vix
    except Exception: return "Neutral", 20.0

def get_vix_term_structure() -> str:
    return "Contango"

# --- CORE DATA & CALCULATION ---
def get_ticker_yf(s: str) -> Optional[yf.Ticker]:
    if s in yf_cache and (time.time() - yf_cache[s]['timestamp'] < 300): return yf_cache[s]['ticker']
    try:
        t = yf.Ticker(s)
        if 'regularMarketPrice' in t.info and t.info['regularMarketPrice'] is not None:
            yf_cache[s] = {'ticker': t, 'timestamp': time.time()}; return t
    except Exception: return None

def get_historical_data_yf(t: yf.Ticker) -> Optional[pd.DataFrame]:
    try:
        h = t.history(period="1y")
        if h.empty: return None
        h.rename(columns=str.capitalize, inplace=True); return h
    except Exception: return None

def get_options_chain_yf(t: yf.Ticker, c: Config, d: int) -> Optional[pd.DataFrame]:
    try:
        expiries = t.options
        if not expiries: return None
        today = datetime.now().date()
        valid = [(e, (datetime.strptime(e, '%Y-%m-%d').date() - today).days) for e in expiries if (datetime.strptime(e, '%Y-%m-%d').date() - today).days > 5]
        if not valid: return None
        expiry, days = min(valid, key=lambda x: abs(x[1] - d))
        chain = t.option_chain(expiry)
        df = pd.concat([chain.calls.assign(option_type='call'), chain.puts.assign(option_type='put')])
        df.rename(columns={'openInterest': 'openInterest', 'impliedVolatility': 'impliedVolatility'}, inplace=True)
        df = df[(df['openInterest'] >= c.min_oi) & (df['bid'] > 0) & (df['ask'] > 0)].copy()
        if df.empty: return None
        df['spread_pct'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
        df = df[df['spread_pct'] <= c.max_bid_ask_spread_pct]
        if df.empty or 'call' not in df['option_type'].unique() or 'put' not in df['option_type'].unique(): return None
        df.loc[:, 'expiry_date'], df.loc[:, 'days_to_exp'] = expiry, days
        return df
    except Exception: return None

def calculate_hv20(h: pd.DataFrame) -> float:
    if len(h) < 21: return np.nan
    return np.log(h['Close']/h['Close'].shift(1)).rolling(20).std().iloc[-1] * sqrt(252) * 100

def calculate_parkinson20(h: pd.DataFrame) -> float:
    if len(h) < 21: return np.nan
    return sqrt((np.log(h['High']/h['Low'])**2).rolling(20).mean().iloc[-1] / (4*log(2))) * sqrt(252) * 100

def calculate_variance_ratio(h: pd.DataFrame) -> Tuple[float, str]:
    if len(h) < 21: return np.nan, "neutral"
    log_returns = np.log(h['Close']/h['Close'].shift(1)).dropna()
    var1 = log_returns.var()
    if var1 == 0: return np.nan, "neutral"
    vr = log_returns.rolling(20).sum().var() / (20 * var1)
    return vr, "trend" if vr > 1.05 else "meanrev" if vr < 0.95 else "neutral"

def get_atm_iv(df: pd.DataFrame, s: float) -> float:
    if df is None or df.empty: return np.nan
    atm_opts = df.iloc[(df['strike'] - s).abs().argsort()[:10]]
    valid_ivs = atm_opts['impliedVolatility'][(atm_opts['impliedVolatility'] > 0.01) & (atm_opts['impliedVolatility'] < 10.0)]
    return valid_ivs.median() if not valid_ivs.empty else np.nan

def score_and_construct_trade(idea: dict, opts: pd.DataFrame, hist: pd.DataFrame) -> Optional[dict]:
    symbol = idea['symbol']
    calls, puts = opts[opts['option_type'] == 'call'], opts[opts['option_type'] == 'put']
    scores = {"TREND_FOLLOW": 0, "MEAN_REVERSION": 0, "NAKED_PREMIUM_SELL": 0}
    signals = []
    if idea['vr_signal'] == "trend": scores["TREND_FOLLOW"] += 1.2; signals.append(f"✅ Trend (VR={idea['vr_20']:.2f})")
    if idea['vr_signal'] == "meanrev": scores["MEAN_REVERSION"] += 1.2; signals.append(f"✅ Mean Reversion (VR={idea['vr_20']:.2f})")
    if idea['iv_hv_gap'] > 5.0: scores["MEAN_REVERSION"] += 1.5; scores["NAKED_PREMIUM_SELL"] += 1.5; signals.append(f"✅ High IV Prem (Gap: {idea['iv_hv_gap']:.1f} pts)")
    if idea['market_regime'] == "High Volatility": scores["MEAN_REVERSION"] *= 1.5; scores["NAKED_PREMIUM_SELL"] *= 1.7
    elif idea['market_regime'] == "Low Volatility": scores["TREND_FOLLOW"] *= 1.5
    logger.info(f"[{symbol}] SCORES -> Trend: {scores['TREND_FOLLOW']:.2f}, Mean Reversion: {scores['MEAN_REVERSION']:.2f}, Naked Sell: {scores['NAKED_PREMIUM_SELL']:.2f}")
    best_strategy = max(scores, key=scores.get)
    if scores[best_strategy] < 1.5: return None
    logger.info(f"[{symbol}] Selected strategy: {best_strategy} with score {scores[best_strategy]:.2f}")
    idea.update({'strategy': best_strategy, 'confluence': signals})
    S, r, T = idea['current_price'], idea['risk_free_rate'], opts['days_to_exp'].iloc[0] / 365
    
    calls, puts = calls.sort_values('strike'), puts.sort_values('strike', ascending=False)
    
    def format_leg(leg, action, opt_type): return f"{action} 1x {idea['symbol']} {idea['expiry_date']} {leg['strike']:.2f} {opt_type} (@ {leg['bid']:.2f}-{leg['ask']:.2f})"
    
    if best_strategy == "TREND_FOLLOW":
        is_uptrend = idea.get('is_uptrend', False)
        idea['rationale'] = f"Low-cost directional bet on continued {'upward' if is_uptrend else 'downward'} momentum."
        try:
            if not is_uptrend: # Bearish Put Debit Spread
                long_leg = puts[puts['strike'] >= S].iloc[-1] # First put ITM/ATM
                short_leg = puts[puts['strike'] < long_leg['strike']].iloc[0]
            else: # Bullish Call Debit Spread
                long_leg = calls[calls['strike'] <= S].iloc[-1] # First call ITM/ATM
                short_leg = calls[calls['strike'] > long_leg['strike']].iloc[0]
            
            d1, t1 = calculate_greeks(long_leg['option_type'], S, long_leg['strike'], T, r, long_leg['impliedVolatility'])
            d2, t2 = calculate_greeks(short_leg['option_type'], S, short_leg['strike'], T, r, short_leg['impliedVolatility'])
            net_delta, net_theta, net_debit = (d1 - d2) * 100, (t1 - t2) * 100, (long_leg['ask'] - short_leg['bid']) * 100
            if net_debit <= 0: return None
            trade = [format_leg(long_leg, "BUY", long_leg['option_type'].title()), format_leg(short_leg, "SELL", short_leg['option_type'].title()), f"NET DEBIT: ${net_debit:.2f}"]
            idea.update({"trade": trade, "max_loss": f"${net_debit:.2f}", "invalidation_point": f"Close {'<' if is_uptrend else '>'} ${long_leg['strike']:.2f}", "exit_strategy": "Exit for profit at +50% of max profit.", "net_delta": net_delta, "net_theta": net_theta, "is_income": False}); return idea
        except IndexError: return None

    elif best_strategy == "NAKED_PREMIUM_SELL":
        idea['rationale'] = "Capital-efficient bet on high volatility and a stable/rising stock price."
        otm_puts = puts[puts['strike'] < S * 0.97] # Find puts ~3% OTM
        if otm_puts.empty: return None
        sp = otm_puts.iloc[0]
        spd, spt = calculate_greeks('p', S, sp['strike'], T, r, sp['impliedVolatility'])
        net_delta, net_theta, net_credit = -spd * 100, -spt * 100, sp['bid'] * 100
        trade = [format_leg(sp, "SELL", "Put"), f"NET CREDIT: ${net_credit:.2f}"]
        idea.update({"trade": trade, "max_loss": "Substantial", "invalidation_point": f"Close < {sp['strike']:.2f}", "exit_strategy": "Close for profit at 50% of credit.", "net_delta": net_delta, "net_theta": net_theta, "is_income": True, "is_naked": True}); return idea
        
    elif best_strategy == "MEAN_REVERSION":
        # DEFINITIVE FIX: Switched to a resilient, index-based strike selection.
        idea['rationale'] = "Risk-defined bet on volatility returning to normal and price staying in a range."
        try:
            otm_puts = puts[puts['strike'] < S]
            otm_calls = calls[calls['strike'] > S]
            
            # Ensure we have enough strikes to build a 4-legged trade
            if len(otm_puts) < 2 or len(otm_calls) < 2: return None
            
            # Sell the first OTM put/call, buy the next one out
            sp, lp = otm_puts.iloc[0], otm_puts.iloc[1]
            sc, lc = otm_calls.iloc[0], otm_calls.iloc[1]

            spd, spt = calculate_greeks('p', S, sp['strike'], T, r, sp['impliedVolatility']); lpd, lpt = calculate_greeks('p', S, lp['strike'], T, r, lp['impliedVolatility'])
            scd, sct = calculate_greeks('c', S, sc['strike'], T, r, sc['impliedVolatility']); lcd, lct = calculate_greeks('c', S, lc['strike'], T, r, lc['impliedVolatility'])
            net_delta, net_theta = (-spd + lpd - scd + lcd) * 100, (-spt + lpt - sct + lct) * 100
            net_credit = (sp['bid'] - lp['ask'] + sc['bid'] - lc['ask']) * 100
            if net_credit <= 0: return None
            trade = [format_leg(sp, "SELL", 'Put'), format_leg(lp, "BUY", 'Put'), format_leg(sc, "SELL", 'Call'), format_leg(lc, "BUY", 'Call'), f"NET CREDIT: ${net_credit:.2f}"]
            idea.update({"trade": trade, "max_loss": f"${(abs(sp['strike'] - lp['strike']) * 100) - net_credit:.2f}", "invalidation_point": f"Close outside {sp['strike']:.2f}-{sc['strike']:.2f}", "exit_strategy": "Close for profit at 50% of credit.", "net_delta": net_delta, "net_theta": net_theta, "is_income": True}); return idea
        except IndexError:
            return None # This will catch errors if there aren't enough strikes.

    return None

def construct_portfolios(ideas: List[dict]) -> List[dict]:
    portfolios, ideas = [], sorted(ideas, key=lambda x: x.get('asym_score', 0), reverse=True)
    if not ideas: return []
    portfolios.append({"title": "All Individual Opportunities", "trades": ideas, "rationale": "The highest conviction trade ideas found during the scan, sorted by their score."})
    naked_sell = next((i for i in ideas if i.get('is_naked')), None)
    bearish_hedge = next((i for i in ideas if not i.get('is_income') and i.get('net_delta', 0) < -10), None)
    if naked_sell and bearish_hedge:
        portfolios.append({"title": "Capital-Efficient Pair Trade", "trades": [naked_sell, bearish_hedge], "rationale": "This portfolio combines a high-probability income trade with a cheap directional hedge."})
    return portfolios

def process_symbol(s: str, c: Config) -> Optional[dict]:
    t = get_ticker_yf(s)
    if not t: return None
    h = get_historical_data_yf(t)
    if h is None or len(h) < 30: return None
    p = h['Close'].iloc[-1]
    if p < c.min_price: return None
    o30, o90 = get_options_chain_yf(t, c, 30), get_options_chain_yf(t, c, 90)
    if o30 is None or o90 is None: return None
    iv30, iv90, hv20, pk20, (vr20, vrsig) = get_atm_iv(o30, p), get_atm_iv(o90, p), calculate_hv20(h), calculate_parkinson20(h), calculate_variance_ratio(h)
    if any(map(pd.isna, [iv30, iv90, hv20, pk20, vr20])): return None
    h['SMA20'] = h['Close'].rolling(20).mean()
    is_uptrend = p > h['SMA20'].iloc[-1]
    idea = {"symbol": s, "current_price": p, "hv20": hv20, "parkinson20": pk20, "vr_20": vr20, "vr_signal": vrsig, "iv30": iv30 * 100, "iv90": iv90 * 100, "term_structure_slope": (iv90 - iv30) * 100, "iv_hv_gap": (iv30 * 100) - hv20, "risk_free_rate": c.risk_free_rate, "expiry_date": o30['expiry_date'].iloc[0], "is_uptrend": is_uptrend}
    idea['asym_score'] = idea['iv_hv_gap'] - idea['term_structure_slope']
    logger.info(f"[{s}] Processed successfully. IV30: {idea['iv30']:.2f}, HV20: {idea['hv20']:.2f}")
    return idea, o30, h

@app.get("/get-market-environment")
async def get_market_env():
    regime, vix = get_market_regime()
    vix_structure = get_vix_term_structure()
    return {"regime": regime, "vix_level": vix, "vix_term_structure": vix_structure}

@app.get("/get-universe")
async def get_uni(): return {"symbols": get_universe()}

@app.post("/update-universe")
async def update_uni(req: UniverseRequest):
    try:
        df = pd.DataFrame(req.symbols, columns=['symbol']); df['symbol'] = df['symbol'].str.strip().str.upper(); df.drop_duplicates(inplace=True); df.to_csv(UNIVERSE_FILE, index=False)
        return {"message": "Universe updated."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan")
async def run_scan():
    logger.info("="*20 + " SCAN REQUEST RECEIVED " + "="*20)
    c, (regime, _), all_ideas = Config(), get_market_regime(), []
    vix_structure = get_vix_term_structure()
    logger.info(f"Market Environment -> Regime: {regime}, VIX Structure: {vix_structure}")
    universe = get_universe()
    if not universe:
        logger.error("Universe is empty. Cannot run scan.")
        return []
    for s in universe:
        logger.info(f"--- Processing {s} ---")
        try:
            processed = process_symbol(s, c)
            if processed:
                base, opts, hist = processed
                base.update({'market_regime': regime})
                trade = score_and_construct_trade(base, opts, hist)
                if trade:
                    logger.info(f"[{s}] >>> TRADE FOUND: {trade['strategy']} <<<")
                    all_ideas.append(trade)
        except Exception as e:
            logger.error(f"[{s}] A critical error occurred during processing: {e}", exc_info=True)
        time.sleep(0.2)
    logger.info("="*20 + " SCAN COMPLETE " + "="*20)
    if not all_ideas:
        logger.warning("Scan finished but no viable trade ideas were found.")
        return []
    return construct_portfolios(all_ideas)