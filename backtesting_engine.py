# Advanced Backtesting Engine for Options Strategies
# Comprehensive historical performance analysis and optimization

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Results from backtesting a strategy"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    trades: List[Dict]
    daily_returns: List[float]
    equity_curve: List[float]
    drawdown_curve: List[float]

class OptionsBacktester:
    """Advanced backtesting engine for options strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.risk_free_rate = 0.045
        self.commission_per_trade = 0.65  # $0.65 per contract
        self.slippage = 0.01  # 1% slippage
        
    def backtest_strategy(self, 
                         strategy_config: Dict, 
                         start_date: str, 
                         end_date: str,
                         symbol: str) -> BacktestResult:
        """Backtest a single strategy"""
        try:
            logger.info(f"Backtesting {strategy_config['name']} for {symbol} from {start_date} to {end_date}")
            
            # Get historical data
            hist_data = self._get_historical_data(symbol, start_date, end_date)
            if hist_data is None or len(hist_data) < 30:
                logger.error(f"Insufficient data for {symbol}")
                return None
            
            # Get options data (simplified - in real implementation would use historical options data)
            options_data = self._get_historical_options_data(symbol, start_date, end_date)
            
            # Initialize backtest
            capital = self.initial_capital
            positions = []
            trades = []
            daily_returns = []
            equity_curve = [capital]
            
            # Simulate strategy execution
            for i in range(30, len(hist_data)):  # Start after 30 days for indicators
                current_date = hist_data.index[i]
                current_price = hist_data['Close'].iloc[i]
                
                # Check for strategy signals
                signal = self._generate_strategy_signal(
                    strategy_config, hist_data.iloc[:i+1], current_price, options_data
                )
                
                if signal and signal['action'] == 'enter':
                    # Enter position
                    position = self._enter_position(
                        strategy_config, current_price, current_date, capital
                    )
                    if position:
                        positions.append(position)
                        capital -= position['cost']
                
                elif signal and signal['action'] == 'exit':
                    # Exit positions
                    for position in positions[:]:
                        exit_price = self._calculate_exit_price(position, current_price)
                        pnl = self._calculate_pnl(position, exit_price)
                        
                        trade = {
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'quantity': position['quantity'],
                            'pnl': pnl,
                            'return_pct': pnl / position['cost'] * 100
                        }
                        trades.append(trade)
                        
                        capital += position['cost'] + pnl
                        positions.remove(position)
                
                # Update daily metrics
                daily_return = (capital - equity_curve[-1]) / equity_curve[-1] if equity_curve else 0
                daily_returns.append(daily_return)
                equity_curve.append(capital)
            
            # Close any remaining positions
            for position in positions:
                final_price = hist_data['Close'].iloc[-1]
                exit_price = self._calculate_exit_price(position, final_price)
                pnl = self._calculate_pnl(position, exit_price)
                
                trade = {
                    'entry_date': position['entry_date'],
                    'exit_date': hist_data.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'return_pct': pnl / position['cost'] * 100
                }
                trades.append(trade)
                
                capital += position['cost'] + pnl
            
            # Calculate performance metrics
            final_capital = capital
            total_return = (final_capital - self.initial_capital) / self.initial_capital
            
            # Annualized return
            days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
            annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # Volatility
            volatility = np.std(daily_returns) * np.sqrt(252)
            
            # Sharpe ratio
            excess_returns = np.array(daily_returns) - self.risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
            
            # Sortino ratio
            downside_returns = [r for r in daily_returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            sortino_ratio = np.mean(excess_returns) / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0
            
            # Max drawdown
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade statistics
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average win/loss
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Largest win/loss
            largest_win = max([t['pnl'] for t in trades]) if trades else 0
            largest_loss = min([t['pnl'] for t in trades]) if trades else 0
            
            return BacktestResult(
                strategy_name=strategy_config['name'],
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                trades=trades,
                daily_returns=daily_returns,
                equity_curve=equity_curve,
                drawdown_curve=drawdown.tolist()
            )
            
        except Exception as e:
            logger.error(f"Error backtesting {strategy_config['name']}: {e}")
            return None
    
    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Add technical indicators
            hist['Returns'] = hist['Close'].pct_change()
            hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            hist['SMA_20'] = hist['Close'].rolling(20).mean()
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['Volatility'] = hist['Log_Returns'].rolling(20).std() * np.sqrt(252)
            
            return hist
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _get_historical_options_data(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Get historical options data (simplified simulation)"""
        # In a real implementation, this would fetch historical options data
        # For now, we'll simulate based on historical volatility
        return {
            'implied_volatility': 0.25,  # Simulated constant IV
            'options_available': True
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _generate_strategy_signal(self, 
                                 strategy_config: Dict, 
                                 hist_data: pd.DataFrame, 
                                 current_price: float,
                                 options_data: Dict) -> Optional[Dict]:
        """Generate trading signals based on strategy configuration"""
        try:
            strategy_type = strategy_config.get('type', 'income')
            
            if strategy_type == 'iron_condor':
                return self._iron_condor_signal(hist_data, current_price, options_data)
            elif strategy_type == 'straddle':
                return self._straddle_signal(hist_data, current_price, options_data)
            elif strategy_type == 'covered_call':
                return self._covered_call_signal(hist_data, current_price, options_data)
            elif strategy_type == 'protective_put':
                return self._protective_put_signal(hist_data, current_price, options_data)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _iron_condor_signal(self, hist_data: pd.DataFrame, current_price: float, options_data: Dict) -> Optional[Dict]:
        """Generate Iron Condor signals"""
        if len(hist_data) < 20:
            return None
        
        # Check if we're in a range-bound market
        recent_high = hist_data['High'].tail(20).max()
        recent_low = hist_data['Low'].tail(20).min()
        range_size = (recent_high - recent_low) / current_price
        
        # Enter if range is small (sideways market)
        if range_size < 0.05:  # 5% range
            return {'action': 'enter', 'strategy': 'iron_condor'}
        
        # Exit if range breaks out
        if current_price > recent_high * 1.02 or current_price < recent_low * 0.98:
            return {'action': 'exit', 'strategy': 'iron_condor'}
        
        return None
    
    def _straddle_signal(self, hist_data: pd.DataFrame, current_price: float, options_data: Dict) -> Optional[Dict]:
        """Generate Straddle signals"""
        if len(hist_data) < 20:
            return None
        
        # Check for low volatility breakout setup
        recent_vol = hist_data['Volatility'].tail(20).mean()
        long_term_vol = hist_data['Volatility'].tail(50).mean()
        
        # Enter if volatility is low and increasing
        if recent_vol < long_term_vol * 0.8 and hist_data['Volatility'].iloc[-1] > recent_vol * 1.1:
            return {'action': 'enter', 'strategy': 'straddle'}
        
        # Exit after significant move
        if len(hist_data) > 0:
            entry_price = hist_data['Close'].iloc[-20]  # Assume entry 20 days ago
            move = abs(current_price - entry_price) / entry_price
            if move > 0.1:  # 10% move
                return {'action': 'exit', 'strategy': 'straddle'}
        
        return None
    
    def _covered_call_signal(self, hist_data: pd.DataFrame, current_price: float, options_data: Dict) -> Optional[Dict]:
        """Generate Covered Call signals"""
        if len(hist_data) < 20:
            return None
        
        # Enter if stock is near resistance
        sma_20 = hist_data['SMA_20'].iloc[-1]
        if current_price > sma_20 * 1.02:  # Above SMA
            return {'action': 'enter', 'strategy': 'covered_call'}
        
        # Exit if stock drops below SMA
        if current_price < sma_20 * 0.98:
            return {'action': 'exit', 'strategy': 'covered_call'}
        
        return None
    
    def _protective_put_signal(self, hist_data: pd.DataFrame, current_price: float, options_data: Dict) -> Optional[Dict]:
        """Generate Protective Put signals"""
        if len(hist_data) < 20:
            return None
        
        # Enter if stock is near support
        sma_20 = hist_data['SMA_20'].iloc[-1]
        if current_price < sma_20 * 0.98:  # Below SMA
            return {'action': 'enter', 'strategy': 'protective_put'}
        
        # Exit if stock recovers
        if current_price > sma_20 * 1.02:
            return {'action': 'exit', 'strategy': 'protective_put'}
        
        return None
    
    def _enter_position(self, strategy_config: Dict, current_price: float, current_date: datetime, capital: float) -> Optional[Dict]:
        """Enter a new position"""
        try:
            position_size = min(capital * 0.1, 10000)  # Max 10% of capital or $10k
            quantity = int(position_size / current_price)
            
            if quantity < 1:
                return None
            
            # Calculate position cost (simplified)
            cost = quantity * current_price * 1.01  # Include slippage
            
            return {
                'strategy': strategy_config['name'],
                'entry_date': current_date,
                'entry_price': current_price,
                'quantity': quantity,
                'cost': cost
            }
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            return None
    
    def _calculate_exit_price(self, position: Dict, current_price: float) -> float:
        """Calculate exit price for a position"""
        # Simplified - in real implementation would calculate based on options pricing
        return current_price
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate profit/loss for a position"""
        # Simplified calculation
        price_change = exit_price - position['entry_price']
        pnl = price_change * position['quantity']
        
        # Subtract commissions
        pnl -= self.commission_per_trade * 2  # Entry and exit
        
        return pnl
    
    def backtest_multiple_strategies(self, 
                                   strategies: List[Dict], 
                                   symbols: List[str],
                                   start_date: str, 
                                   end_date: str) -> List[BacktestResult]:
        """Backtest multiple strategies across multiple symbols"""
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for strategy in strategies:
                for symbol in symbols:
                    future = executor.submit(
                        self.backtest_strategy, 
                        strategy, 
                        start_date, 
                        end_date, 
                        symbol
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
    def optimize_strategy_parameters(self, 
                                   strategy_config: Dict, 
                                   symbol: str,
                                   start_date: str, 
                                   end_date: str,
                                   parameter_ranges: Dict) -> Dict:
        """Optimize strategy parameters using grid search"""
        best_result = None
        best_params = None
        best_score = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        for params in param_combinations:
            # Update strategy config with new parameters
            test_config = strategy_config.copy()
            test_config.update(params)
            
            # Run backtest
            result = self.backtest_strategy(test_config, start_date, end_date, symbol)
            
            if result:
                # Score based on Sharpe ratio and total return
                score = result.sharpe_ratio * 0.7 + result.total_return * 0.3
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_params = params
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'best_score': best_score
        }
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """Generate all combinations of parameters for optimization"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
        
        return combinations
    
    def generate_performance_report(self, results: List[BacktestResult]) -> Dict:
        """Generate comprehensive performance report"""
        if not results:
            return {}
        
        # Aggregate statistics
        total_strategies = len(results)
        avg_return = np.mean([r.total_return for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_max_dd = np.mean([r.max_drawdown for r in results])
        
        # Best and worst performers
        best_strategy = max(results, key=lambda r: r.total_return)
        worst_strategy = min(results, key=lambda r: r.total_return)
        
        # Risk metrics
        returns = [r.total_return for r in results]
        volatility = np.std(returns)
        
        # Win rate
        profitable_strategies = len([r for r in results if r.total_return > 0])
        win_rate = profitable_strategies / total_strategies
        
        return {
            'total_strategies': total_strategies,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_max_drawdown': avg_max_dd,
            'volatility': volatility,
            'win_rate': win_rate,
            'best_strategy': {
                'name': best_strategy.strategy_name,
                'symbol': best_strategy.symbol,
                'return': best_strategy.total_return,
                'sharpe': best_strategy.sharpe_ratio
            },
            'worst_strategy': {
                'name': worst_strategy.strategy_name,
                'symbol': worst_strategy.symbol,
                'return': worst_strategy.total_return,
                'sharpe': worst_strategy.sharpe_ratio
            }
        }
    
    def create_performance_charts(self, results: List[BacktestResult]) -> Dict[str, go.Figure]:
        """Create performance visualization charts"""
        charts = {}
        
        if not results:
            return charts
        
        # Returns distribution
        returns = [r.total_return * 100 for r in results]
        fig_returns = go.Figure(data=[go.Histogram(x=returns, nbinsx=20)])
        fig_returns.update_layout(
            title="Returns Distribution",
            xaxis_title="Total Return (%)",
            yaxis_title="Frequency"
        )
        charts['returns_distribution'] = fig_returns
        
        # Risk vs Return scatter
        returns_scatter = [r.total_return * 100 for r in results]
        volatilities = [r.volatility * 100 for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        
        fig_risk_return = go.Figure(data=go.Scatter(
            x=volatilities,
            y=returns_scatter,
            mode='markers',
            text=[f"{r.strategy_name} ({r.symbol})" for r in results],
            marker=dict(
                size=8,
                color=sharpe_ratios,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            )
        ))
        fig_risk_return.update_layout(
            title="Risk vs Return",
            xaxis_title="Volatility (%)",
            yaxis_title="Total Return (%)"
        )
        charts['risk_return'] = fig_risk_return
        
        # Equity curves for top performers
        top_5 = sorted(results, key=lambda r: r.total_return, reverse=True)[:5]
        
        fig_equity = go.Figure()
        for result in top_5:
            fig_equity.add_trace(go.Scatter(
                x=list(range(len(result.equity_curve))),
                y=result.equity_curve,
                mode='lines',
                name=f"{result.strategy_name} ({result.symbol})"
            ))
        
        fig_equity.update_layout(
            title="Equity Curves - Top 5 Strategies",
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)"
        )
        charts['equity_curves'] = fig_equity
        
        return charts

# Example usage and testing
if __name__ == "__main__":
    # Initialize backtester
    backtester = OptionsBacktester(initial_capital=100000)
    
    # Define strategies to test
    strategies = [
        {
            'name': 'Iron Condor',
            'type': 'iron_condor',
            'max_risk': 0.02
        },
        {
            'name': 'Straddle',
            'type': 'straddle',
            'max_risk': 0.05
        },
        {
            'name': 'Covered Call',
            'type': 'covered_call',
            'max_risk': 0.01
        }
    ]
    
    # Test symbols
    symbols = ['AAPL', 'TSLA', 'GOOGL', 'NVDA']
    
    # Run backtests
    results = backtester.backtest_multiple_strategies(
        strategies=strategies,
        symbols=symbols,
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # Generate report
    report = backtester.generate_performance_report(results)
    print("Performance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Create charts
    charts = backtester.create_performance_charts(results)
    print(f"Generated {len(charts)} performance charts")