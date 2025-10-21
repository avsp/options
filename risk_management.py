# Advanced Risk Management System for Options Trading
# Comprehensive risk controls, position sizing, and portfolio protection

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum
import json
import math
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class AlertType(Enum):
    POSITION_SIZE = "position_size"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    MARGIN = "margin"
    CONCENTRATION = "concentration"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_type: AlertType
    severity: RiskLevel
    message: str
    current_value: float
    threshold_value: float
    recommendation: str
    timestamp: datetime

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    strategy: str
    position_size: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    var_95: float
    var_99: float
    max_loss: float
    beta: float
    correlation_to_spy: float
    risk_score: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    portfolio_var_95: float
    portfolio_var_99: float
    max_drawdown: float
    portfolio_beta: float
    diversification_ratio: float
    concentration_risk: float
    correlation_risk: float
    volatility_risk: float
    overall_risk_score: float
    margin_requirement: float
    available_capital: float
    risk_capacity: float

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,
                 max_position_risk: float = 0.05,
                 max_correlation: float = 0.7,
                 max_concentration: float = 0.2,
                 max_drawdown: float = 0.1,
                 initial_capital: float = 100000):
        
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.max_concentration = max_concentration
        self.max_drawdown = max_drawdown
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.risk_alerts = []
        self.risk_history = []
        
        # Risk thresholds
        self.thresholds = {
            'position_size_pct': 0.05,  # 5% max position size
            'portfolio_var_95': 0.02,   # 2% daily VaR
            'portfolio_var_99': 0.05,   # 5% daily VaR
            'max_correlation': 0.7,     # 70% max correlation
            'concentration_limit': 0.2,  # 20% max concentration
            'drawdown_limit': 0.1,      # 10% max drawdown
            'volatility_limit': 0.3,    # 30% max portfolio volatility
            'margin_utilization': 0.8   # 80% max margin utilization
        }
    
    def add_position(self, position: Dict) -> bool:
        """Add a new position with risk validation"""
        try:
            # Calculate position risk metrics
            position_risk = self._calculate_position_risk(position)
            
            # Validate position against risk limits
            validation_result = self._validate_position(position_risk)
            
            if not validation_result['valid']:
                # Create risk alert
                alert = RiskAlert(
                    alert_type=AlertType.POSITION_SIZE,
                    severity=RiskLevel.HIGH,
                    message=f"Position rejected: {validation_result['reason']}",
                    current_value=position_risk.position_size,
                    threshold_value=self.max_position_risk,
                    recommendation=validation_result['recommendation'],
                    timestamp=datetime.now()
                )
                self.risk_alerts.append(alert)
                return False
            
            # Add position
            self.positions.append({
                'position': position,
                'risk': position_risk,
                'entry_time': datetime.now()
            })
            
            # Update portfolio risk
            self._update_portfolio_risk()
            
            logger.info(f"Position added: {position['symbol']} - {position['strategy']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, symbol: str, strategy: str) -> bool:
        """Remove a position"""
        try:
            for i, pos in enumerate(self.positions):
                if (pos['position']['symbol'] == symbol and 
                    pos['position']['strategy'] == strategy):
                    del self.positions[i]
                    self._update_portfolio_risk()
                    logger.info(f"Position removed: {symbol} - {strategy}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Error removing position: {e}")
            return False
    
    def _calculate_position_risk(self, position: Dict) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position"""
        try:
            # Basic metrics
            symbol = position['symbol']
            strategy = position['strategy']
            position_size = position.get('position_size', 0)
            
            # Greeks
            delta = position.get('delta', 0)
            gamma = position.get('gamma', 0)
            theta = position.get('theta', 0)
            vega = position.get('vega', 0)
            rho = position.get('rho', 0)
            
            # Calculate VaR (simplified)
            volatility = position.get('volatility', 0.2)
            var_95 = position_size * volatility * 1.645  # 95% VaR
            var_99 = position_size * volatility * 2.326  # 99% VaR
            
            # Max loss
            max_loss = position.get('max_loss', position_size)
            
            # Beta and correlation (simplified)
            beta = position.get('beta', 1.0)
            correlation_to_spy = position.get('correlation_to_spy', 0.5)
            
            # Risk score (0-1, higher is riskier)
            risk_score = self._calculate_risk_score(
                position_size, delta, gamma, theta, vega, 
                var_95, max_loss, beta, correlation_to_spy
            )
            
            return PositionRisk(
                symbol=symbol,
                strategy=strategy,
                position_size=position_size,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                var_95=var_95,
                var_99=var_99,
                max_loss=max_loss,
                beta=beta,
                correlation_to_spy=correlation_to_spy,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return PositionRisk(symbol="", strategy="", position_size=0, delta=0, gamma=0, 
                              theta=0, vega=0, rho=0, var_95=0, var_99=0, max_loss=0, 
                              beta=0, correlation_to_spy=0, risk_score=1.0)
    
    def _calculate_risk_score(self, position_size: float, delta: float, gamma: float, 
                            theta: float, vega: float, var_95: float, max_loss: float,
                            beta: float, correlation: float) -> float:
        """Calculate overall risk score for a position"""
        try:
            # Size risk (0-1)
            size_risk = min(1.0, position_size / (self.current_capital * self.max_position_risk))
            
            # Greeks risk (0-1)
            delta_risk = min(1.0, abs(delta) / 100)  # Normalize delta
            gamma_risk = min(1.0, abs(gamma) / 10)   # Normalize gamma
            theta_risk = min(1.0, abs(theta) / 10)   # Normalize theta
            vega_risk = min(1.0, abs(vega) / 100)    # Normalize vega
            
            # VaR risk (0-1)
            var_risk = min(1.0, var_95 / (self.current_capital * 0.01))
            
            # Loss risk (0-1)
            loss_risk = min(1.0, max_loss / (self.current_capital * 0.1))
            
            # Beta risk (0-1)
            beta_risk = min(1.0, abs(beta - 1.0) / 2.0)
            
            # Correlation risk (0-1)
            correlation_risk = min(1.0, abs(correlation))
            
            # Weighted average
            weights = {
                'size': 0.25,
                'delta': 0.15,
                'gamma': 0.10,
                'theta': 0.10,
                'vega': 0.10,
                'var': 0.15,
                'loss': 0.10,
                'beta': 0.05
            }
            
            risk_score = (
                size_risk * weights['size'] +
                delta_risk * weights['delta'] +
                gamma_risk * weights['gamma'] +
                theta_risk * weights['theta'] +
                vega_risk * weights['vega'] +
                var_risk * weights['var'] +
                loss_risk * weights['loss'] +
                beta_risk * weights['beta']
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 1.0
    
    def _validate_position(self, position_risk: PositionRisk) -> Dict:
        """Validate position against risk limits"""
        try:
            # Check position size
            if position_risk.position_size > self.current_capital * self.max_position_risk:
                return {
                    'valid': False,
                    'reason': f"Position size {position_risk.position_size:.2f} exceeds limit",
                    'recommendation': f"Reduce position size to {self.current_capital * self.max_position_risk:.2f}"
                }
            
            # Check VaR
            if position_risk.var_95 > self.current_capital * self.thresholds['portfolio_var_95']:
                return {
                    'valid': False,
                    'reason': f"VaR {position_risk.var_95:.2f} exceeds limit",
                    'recommendation': "Reduce position size or hedge risk"
                }
            
            # Check risk score
            if position_risk.risk_score > 0.8:
                return {
                    'valid': False,
                    'reason': f"Risk score {position_risk.risk_score:.2f} too high",
                    'recommendation': "Reduce position complexity or size"
                }
            
            return {'valid': True, 'reason': '', 'recommendation': ''}
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return {'valid': False, 'reason': 'Validation error', 'recommendation': 'Contact support'}
    
    def _update_portfolio_risk(self):
        """Update portfolio-level risk metrics"""
        try:
            if not self.positions:
                self.portfolio_risk = PortfolioRisk(
                    total_delta=0, total_gamma=0, total_theta=0, total_vega=0,
                    portfolio_var_95=0, portfolio_var_99=0, max_drawdown=0,
                    portfolio_beta=0, diversification_ratio=0, concentration_risk=0,
                    correlation_risk=0, volatility_risk=0, overall_risk_score=0,
                    margin_requirement=0, available_capital=self.current_capital,
                    risk_capacity=self.current_capital * self.max_portfolio_risk
                )
                return
            
            # Aggregate Greeks
            total_delta = sum(pos['risk'].delta for pos in self.positions)
            total_gamma = sum(pos['risk'].gamma for pos in self.positions)
            total_theta = sum(pos['risk'].theta for pos in self.positions)
            total_vega = sum(pos['risk'].vega for pos in self.positions)
            
            # Portfolio VaR (simplified)
            individual_vars = [pos['risk'].var_95 for pos in self.positions]
            portfolio_var_95 = math.sqrt(sum(var**2 for var in individual_vars))  # Assuming independence
            portfolio_var_99 = math.sqrt(sum(pos['risk'].var_99**2 for pos in self.positions))
            
            # Max drawdown (simplified)
            max_losses = [pos['risk'].max_loss for pos in self.positions]
            max_drawdown = sum(max_losses) / self.current_capital
            
            # Portfolio beta (weighted average)
            total_position_size = sum(pos['risk'].position_size for pos in self.positions)
            if total_position_size > 0:
                portfolio_beta = sum(
                    pos['risk'].beta * pos['risk'].position_size 
                    for pos in self.positions
                ) / total_position_size
            else:
                portfolio_beta = 0
            
            # Diversification ratio
            unique_symbols = len(set(pos['position']['symbol'] for pos in self.positions))
            unique_strategies = len(set(pos['position']['strategy'] for pos in self.positions))
            diversification_ratio = (unique_symbols + unique_strategies) / max(1, len(self.positions))
            
            # Concentration risk
            position_sizes = [pos['risk'].position_size for pos in self.positions]
            if position_sizes:
                max_concentration = max(position_sizes) / sum(position_sizes)
                concentration_risk = max_concentration
            else:
                concentration_risk = 0
            
            # Correlation risk (simplified)
            correlations = [pos['risk'].correlation_to_spy for pos in self.positions]
            avg_correlation = np.mean(correlations) if correlations else 0
            correlation_risk = avg_correlation
            
            # Volatility risk
            risk_scores = [pos['risk'].risk_score for pos in self.positions]
            volatility_risk = np.std(risk_scores) if len(risk_scores) > 1 else 0
            
            # Overall risk score
            overall_risk_score = self._calculate_portfolio_risk_score(
                portfolio_var_95, max_drawdown, concentration_risk, 
                correlation_risk, volatility_risk, diversification_ratio
            )
            
            # Margin requirement (simplified)
            margin_requirement = sum(pos['risk'].position_size * 0.1 for pos in self.positions)
            
            # Available capital
            used_capital = sum(pos['risk'].position_size for pos in self.positions)
            available_capital = self.current_capital - used_capital
            
            self.portfolio_risk = PortfolioRisk(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                portfolio_var_95=portfolio_var_95,
                portfolio_var_99=portfolio_var_99,
                max_drawdown=max_drawdown,
                portfolio_beta=portfolio_beta,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                volatility_risk=volatility_risk,
                overall_risk_score=overall_risk_score,
                margin_requirement=margin_requirement,
                available_capital=available_capital,
                risk_capacity=self.current_capital * self.max_portfolio_risk
            )
            
            # Check for risk alerts
            self._check_risk_alerts()
            
        except Exception as e:
            logger.error(f"Error updating portfolio risk: {e}")
    
    def _calculate_portfolio_risk_score(self, var_95: float, max_drawdown: float,
                                      concentration: float, correlation: float,
                                      volatility: float, diversification: float) -> float:
        """Calculate overall portfolio risk score"""
        try:
            # VaR risk (0-1)
            var_risk = min(1.0, var_95 / (self.current_capital * 0.02))
            
            # Drawdown risk (0-1)
            drawdown_risk = min(1.0, max_drawdown / self.max_drawdown)
            
            # Concentration risk (0-1)
            concentration_risk = min(1.0, concentration / self.max_concentration)
            
            # Correlation risk (0-1)
            correlation_risk = min(1.0, correlation / self.max_correlation)
            
            # Volatility risk (0-1)
            volatility_risk = min(1.0, volatility / 0.5)
            
            # Diversification benefit (0-1, inverted)
            diversification_risk = max(0.0, 1.0 - diversification)
            
            # Weighted average
            weights = {
                'var': 0.25,
                'drawdown': 0.20,
                'concentration': 0.15,
                'correlation': 0.15,
                'volatility': 0.15,
                'diversification': 0.10
            }
            
            risk_score = (
                var_risk * weights['var'] +
                drawdown_risk * weights['drawdown'] +
                concentration_risk * weights['concentration'] +
                correlation_risk * weights['correlation'] +
                volatility_risk * weights['volatility'] +
                diversification_risk * weights['diversification']
            )
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk score: {e}")
            return 1.0
    
    def _check_risk_alerts(self):
        """Check for risk threshold violations and create alerts"""
        try:
            if not hasattr(self, 'portfolio_risk'):
                return
            
            pr = self.portfolio_risk
            
            # VaR alerts
            if pr.portfolio_var_95 > self.current_capital * self.thresholds['portfolio_var_95']:
                self._create_alert(
                    AlertType.POSITION_SIZE,
                    RiskLevel.HIGH,
                    f"Portfolio VaR 95% exceeds limit: {pr.portfolio_var_95:.2f}",
                    pr.portfolio_var_95,
                    self.current_capital * self.thresholds['portfolio_var_95'],
                    "Reduce position sizes or add hedges"
                )
            
            # Drawdown alerts
            if pr.max_drawdown > self.max_drawdown:
                self._create_alert(
                    AlertType.DRAWDOWN,
                    RiskLevel.EXTREME,
                    f"Max drawdown exceeds limit: {pr.max_drawdown:.2%}",
                    pr.max_drawdown,
                    self.max_drawdown,
                    "Close high-risk positions immediately"
                )
            
            # Concentration alerts
            if pr.concentration_risk > self.max_concentration:
                self._create_alert(
                    AlertType.CONCENTRATION,
                    RiskLevel.MEDIUM,
                    f"Concentration risk too high: {pr.concentration_risk:.2%}",
                    pr.concentration_risk,
                    self.max_concentration,
                    "Diversify positions across more symbols"
                )
            
            # Correlation alerts
            if pr.correlation_risk > self.max_correlation:
                self._create_alert(
                    AlertType.CORRELATION,
                    RiskLevel.MEDIUM,
                    f"Correlation risk too high: {pr.correlation_risk:.2%}",
                    pr.correlation_risk,
                    self.max_correlation,
                    "Add uncorrelated positions or hedges"
                )
            
            # Overall risk score alerts
            if pr.overall_risk_score > 0.8:
                self._create_alert(
                    AlertType.VOLATILITY,
                    RiskLevel.HIGH,
                    f"Overall risk score too high: {pr.overall_risk_score:.2f}",
                    pr.overall_risk_score,
                    0.8,
                    "Reduce portfolio risk immediately"
                )
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    def _create_alert(self, alert_type: AlertType, severity: RiskLevel, 
                     message: str, current_value: float, threshold_value: float,
                     recommendation: str):
        """Create a risk alert"""
        alert = RiskAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
        self.risk_alerts.append(alert)
        logger.warning(f"Risk Alert: {message}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        try:
            if not hasattr(self, 'portfolio_risk'):
                return {'error': 'No portfolio risk calculated'}
            
            pr = self.portfolio_risk
            
            # Active alerts
            active_alerts = [alert for alert in self.risk_alerts 
                           if (datetime.now() - alert.timestamp).days < 1]
            
            # Risk level
            if pr.overall_risk_score < 0.3:
                risk_level = RiskLevel.LOW
            elif pr.overall_risk_score < 0.6:
                risk_level = RiskLevel.MEDIUM
            elif pr.overall_risk_score < 0.8:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.EXTREME
            
            return {
                'risk_level': risk_level.value,
                'overall_risk_score': pr.overall_risk_score,
                'portfolio_metrics': {
                    'total_delta': pr.total_delta,
                    'total_gamma': pr.total_gamma,
                    'total_theta': pr.total_theta,
                    'total_vega': pr.total_vega,
                    'portfolio_var_95': pr.portfolio_var_95,
                    'portfolio_var_99': pr.portfolio_var_99,
                    'max_drawdown': pr.max_drawdown,
                    'portfolio_beta': pr.portfolio_beta,
                    'diversification_ratio': pr.diversification_ratio,
                    'concentration_risk': pr.concentration_risk,
                    'correlation_risk': pr.correlation_risk,
                    'volatility_risk': pr.volatility_risk
                },
                'capital_metrics': {
                    'available_capital': pr.available_capital,
                    'risk_capacity': pr.risk_capacity,
                    'margin_requirement': pr.margin_requirement,
                    'capital_utilization': (self.current_capital - pr.available_capital) / self.current_capital
                },
                'active_alerts': len(active_alerts),
                'alerts': [
                    {
                        'type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'recommendation': alert.recommendation,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in active_alerts
                ],
                'position_count': len(self.positions),
                'recommendations': self._generate_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if not hasattr(self, 'portfolio_risk'):
            return recommendations
        
        pr = self.portfolio_risk
        
        # Diversification recommendations
        if pr.diversification_ratio < 0.5:
            recommendations.append("Increase portfolio diversification by adding more symbols and strategies")
        
        # Concentration recommendations
        if pr.concentration_risk > 0.15:
            recommendations.append("Reduce concentration risk by limiting position sizes")
        
        # Correlation recommendations
        if pr.correlation_risk > 0.6:
            recommendations.append("Add uncorrelated positions or hedges to reduce correlation risk")
        
        # Volatility recommendations
        if pr.volatility_risk > 0.3:
            recommendations.append("Consider adding volatility hedges or reducing position sizes")
        
        # Capital recommendations
        if pr.available_capital < self.current_capital * 0.1:
            recommendations.append("Maintain higher cash reserves for opportunities and risk management")
        
        # Overall risk recommendations
        if pr.overall_risk_score > 0.7:
            recommendations.append("Portfolio risk is elevated - consider reducing position sizes or adding hedges")
        
        return recommendations
    
    def optimize_position_sizes(self, target_risk: float = None) -> Dict:
        """Optimize position sizes to achieve target risk level"""
        try:
            if target_risk is None:
                target_risk = self.max_portfolio_risk
            
            if not self.positions:
                return {'error': 'No positions to optimize'}
            
            # Current position sizes
            current_sizes = [pos['risk'].position_size for pos in self.positions]
            current_risk = sum(pos['risk'].var_95 for pos in self.positions)
            
            # Calculate scaling factor
            if current_risk > 0:
                scale_factor = (self.current_capital * target_risk) / current_risk
                scale_factor = min(1.0, scale_factor)  # Don't increase sizes
            else:
                scale_factor = 1.0
            
            # Calculate new sizes
            new_sizes = [size * scale_factor for size in current_sizes]
            
            # Calculate new risk metrics
            new_risk = current_risk * scale_factor
            new_var_95 = new_risk
            
            return {
                'current_sizes': current_sizes,
                'new_sizes': new_sizes,
                'scale_factor': scale_factor,
                'current_risk': current_risk,
                'new_risk': new_risk,
                'risk_reduction': (current_risk - new_risk) / current_risk if current_risk > 0 else 0,
                'recommendation': f"Scale positions by {scale_factor:.2f} to achieve target risk"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing position sizes: {e}")
            return {'error': str(e)}
    
    def stress_test(self, scenarios: List[Dict]) -> Dict:
        """Perform stress testing on current portfolio"""
        try:
            if not self.positions:
                return {'error': 'No positions to stress test'}
            
            results = {}
            
            for scenario in scenarios:
                scenario_name = scenario['name']
                market_move = scenario['market_move']  # e.g., -0.1 for -10%
                volatility_change = scenario.get('volatility_change', 0)  # e.g., 0.5 for +50%
                
                # Calculate scenario impact
                total_pnl = 0
                position_impacts = []
                
                for pos in self.positions:
                    # Simplified P&L calculation
                    delta_pnl = pos['risk'].delta * market_move * pos['risk'].position_size
                    vega_pnl = pos['risk'].vega * volatility_change * pos['risk'].position_size
                    position_pnl = delta_pnl + vega_pnl
                    
                    total_pnl += position_pnl
                    position_impacts.append({
                        'symbol': pos['position']['symbol'],
                        'strategy': pos['position']['strategy'],
                        'pnl': position_pnl,
                        'pnl_pct': position_pnl / pos['risk'].position_size if pos['risk'].position_size > 0 else 0
                    })
                
                results[scenario_name] = {
                    'total_pnl': total_pnl,
                    'total_pnl_pct': total_pnl / self.current_capital,
                    'position_impacts': position_impacts
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stress test: {e}")
            return {'error': str(e)}
    
    def export_risk_report(self, filename: str = None) -> str:
        """Export comprehensive risk report"""
        try:
            if filename is None:
                filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'risk_summary': self.get_risk_summary(),
                'positions': [
                    {
                        'symbol': pos['position']['symbol'],
                        'strategy': pos['position']['strategy'],
                        'position_size': pos['risk'].position_size,
                        'risk_score': pos['risk'].risk_score,
                        'delta': pos['risk'].delta,
                        'gamma': pos['risk'].gamma,
                        'theta': pos['risk'].theta,
                        'vega': pos['risk'].vega,
                        'var_95': pos['risk'].var_95,
                        'max_loss': pos['risk'].max_loss
                    }
                    for pos in self.positions
                ],
                'risk_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'recommendation': alert.recommendation,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.risk_alerts
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Risk report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(
        max_portfolio_risk=0.02,
        max_position_risk=0.05,
        initial_capital=100000
    )
    
    # Example positions
    positions = [
        {
            'symbol': 'AAPL',
            'strategy': 'Iron Condor',
            'position_size': 5000,
            'delta': 10,
            'gamma': 2,
            'theta': -5,
            'vega': 15,
            'rho': 1,
            'volatility': 0.25,
            'max_loss': 2000,
            'beta': 1.2,
            'correlation_to_spy': 0.7
        },
        {
            'symbol': 'TSLA',
            'strategy': 'Straddle',
            'position_size': 3000,
            'delta': 0,
            'gamma': 5,
            'theta': -10,
            'vega': 25,
            'rho': 0,
            'volatility': 0.4,
            'max_loss': 3000,
            'beta': 1.5,
            'correlation_to_spy': 0.6
        }
    ]
    
    # Add positions
    for pos in positions:
        risk_manager.add_position(pos)
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print("Risk Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Stress test
    scenarios = [
        {'name': 'Market Crash -10%', 'market_move': -0.1, 'volatility_change': 0.5},
        {'name': 'Market Rally +10%', 'market_move': 0.1, 'volatility_change': -0.2},
        {'name': 'Volatility Spike', 'market_move': 0.0, 'volatility_change': 1.0}
    ]
    
    stress_results = risk_manager.stress_test(scenarios)
    print("\nStress Test Results:")
    print(json.dumps(stress_results, indent=2, default=str))