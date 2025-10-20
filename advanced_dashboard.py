# Advanced Options Trading Dashboard - Alpha Pro Max
# Comprehensive frontend with real-time analytics, strategy visualization, and risk management

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import asyncio
import threading
import time

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app.title = "Advanced Options Strategy Finder - Alpha Pro Max"

# API Configuration
API_BASE_URL = "http://127.0.0.1:8001"

# Global state
current_portfolios = []
market_regime = {}
scan_in_progress = False

# --- UTILITY FUNCTIONS ---
def fetch_api(endpoint, method="GET", data=None):
    """Fetch data from API with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"API Request Error: {e}")
        return None

def format_currency(value):
    """Format currency values"""
    if value is None or pd.isna(value):
        return "$0.00"
    return f"${value:,.2f}"

def format_percentage(value):
    """Format percentage values"""
    if value is None or pd.isna(value):
        return "0.00%"
    return f"{value:.2f}%"

def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        'low': '#10B981',      # Green
        'medium': '#F59E0B',   # Yellow
        'high': '#EF4444',     # Red
        'extreme': '#7C2D12'   # Dark Red
    }
    return colors.get(risk_level, '#6B7280')

# --- LAYOUT COMPONENTS ---
def create_header():
    """Create the main header"""
    return dbc.NavbarSimple(
        brand="Advanced Options Strategy Finder - Alpha Pro Max",
        brand_href="#",
        color="dark",
        dark=True,
        className="mb-4"
    )

def create_market_dashboard():
    """Create market environment dashboard"""
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Market Environment Dashboard", className="mb-0"),
            dbc.Button("Refresh", id="refresh-market-btn", color="primary", size="sm", className="float-end")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Market Regime", className="card-title"),
                            html.H4(id="market-regime", className="text-center"),
                            html.P(id="regime-confidence", className="text-muted text-center")
                        ])
                    ], color="light", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("VIX Level", className="card-title"),
                            html.H4(id="vix-level", className="text-center"),
                            html.P(id="vix-percentile", className="text-muted text-center")
                        ])
                    ], color="light", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("VIX Term Structure", className="card-title"),
                            html.H4(id="vix-term-structure", className="text-center"),
                            html.P("Volatility Curve", className="text-muted text-center")
                        ])
                    ], color="light", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Trend Strength", className="card-title"),
                            html.H4(id="trend-strength", className="text-center"),
                            html.P(id="correlation-regime", className="text-muted text-center")
                        ])
                    ], color="light", outline=True)
                ], width=3)
            ])
        ])
    ])

def create_controls_panel():
    """Create controls panel"""
    return dbc.Card([
        dbc.CardHeader(html.H4("Strategy Controls", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Max Risk per Trade:"),
                    dcc.Slider(
                        id="max-risk-slider",
                        min=0.01,
                        max=0.05,
                        step=0.005,
                        value=0.02,
                        marks={i/100: f"{i}%" for i in range(1, 6, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Min Expected Return:"),
                    dcc.Slider(
                        id="min-return-slider",
                        min=0.05,
                        max=0.30,
                        step=0.05,
                        value=0.15,
                        marks={i/100: f"{i}%" for i in range(5, 31, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], width=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Strategy Types:"),
                    dcc.Checklist(
                        id="strategy-types",
                        options=[
                            {"label": "Income", "value": "income"},
                            {"label": "Directional", "value": "directional"},
                            {"label": "Volatility", "value": "volatility"},
                            {"label": "Arbitrage", "value": "arbitrage"}
                        ],
                        value=["income", "directional", "volatility"],
                        inline=True
                    )
                ], width=8),
                dbc.Col([
                    html.Label("Max Positions:"),
                    dcc.Input(
                        id="max-positions",
                        type="number",
                        min=1,
                        max=20,
                        value=10,
                        className="form-control"
                    )
                ], width=4)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Run Advanced Scan",
                        id="run-scan-btn",
                        color="primary",
                        size="lg",
                        className="w-100"
                    )
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Clear Results",
                        id="clear-results-btn",
                        color="secondary",
                        size="lg",
                        className="w-100"
                    )
                ], width=6)
            ])
        ])
    ])

def create_portfolio_cards(portfolios):
    """Create portfolio cards"""
    if not portfolios:
        return html.Div([
            dbc.Alert("No portfolios found. Run a scan to discover strategies.", color="info")
        ])
    
    cards = []
    for i, portfolio in enumerate(portfolios):
        # Calculate portfolio metrics
        total_strategies = len(portfolio.get('strategies', []))
        total_delta = portfolio.get('total_delta', 0)
        total_theta = portfolio.get('total_theta', 0)
        expected_return = portfolio.get('expected_return', 0)
        risk_score = portfolio.get('risk_score', 0)
        
        # Determine risk color
        if risk_score < 0.3:
            risk_color = "success"
        elif risk_score < 0.6:
            risk_color = "warning"
        else:
            risk_color = "danger"
        
        card = dbc.Card([
            dbc.CardHeader([
                html.H5(f"Portfolio {i+1}", className="mb-0"),
                dbc.Badge(f"{total_strategies} Strategies", color="info", className="ms-2")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Expected Return"),
                        html.H4(format_percentage(expected_return * 100), className="text-success")
                    ], width=3),
                    dbc.Col([
                        html.H6("Net Delta"),
                        html.H4(f"{total_delta:.0f}", className="text-primary")
                    ], width=3),
                    dbc.Col([
                        html.H6("Net Theta"),
                        html.H4(f"{total_theta:.2f}", className="text-info")
                    ], width=3),
                    dbc.Col([
                        html.H6("Risk Score"),
                        html.H4(risk_score, className=f"text-{risk_color}")
                    ], width=3)
                ]),
                html.Hr(),
                dbc.Button(
                    "View Details",
                    id={"type": "portfolio-details-btn", "index": i},
                    color="outline-primary",
                    className="w-100"
                )
            ])
        ], className="mb-3")
        cards.append(card)
    
    return html.Div(cards)

def create_strategy_table(strategies):
    """Create detailed strategy table"""
    if not strategies:
        return html.Div("No strategies available")
    
    # Prepare data for table
    table_data = []
    for strategy in strategies:
        table_data.append({
            "Strategy": strategy.get('name', 'Unknown'),
            "Type": strategy.get('strategy_type', 'Unknown'),
            "Symbol": strategy.get('symbol', 'N/A'),
            "Expected Return": format_percentage(strategy.get('expected_return', 0) * 100),
            "Max Loss": format_currency(strategy.get('max_loss', 0)),
            "Probability of Profit": format_percentage(strategy.get('probability_of_profit', 0) * 100),
            "Risk Level": strategy.get('risk_level', 'Unknown'),
            "Confidence": format_percentage(strategy.get('confidence_score', 0) * 100),
            "Delta": f"{strategy.get('greeks', {}).get('delta', 0):.2f}",
            "Theta": f"{strategy.get('greeks', {}).get('theta', 0):.2f}",
            "Vega": f"{strategy.get('greeks', {}).get('vega', 0):.2f}"
        })
    
    return dash_table.DataTable(
        data=table_data,
        columns=[{"name": col, "id": col} for col in table_data[0].keys()],
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{Risk Level} = high'},
                'backgroundColor': '#ffebee',
                'color': 'black',
            },
            {
                'if': {'filter_query': '{Risk Level} = extreme'},
                'backgroundColor': '#ffcdd2',
                'color': 'black',
            }
        ],
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_current=0,
        page_size=10
    )

def create_risk_metrics_chart(portfolios):
    """Create risk metrics visualization"""
    if not portfolios:
        return go.Figure()
    
    # Extract risk metrics
    portfolio_names = [f"Portfolio {i+1}" for i in range(len(portfolios))]
    risk_scores = [p.get('risk_score', 0) for p in portfolios]
    expected_returns = [p.get('expected_return', 0) * 100 for p in portfolios]
    max_drawdowns = [p.get('max_drawdown', 0) * 100 for p in portfolios]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk vs Return', 'Risk Score Distribution', 'Max Drawdown', 'Portfolio Comparison'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Risk vs Return scatter
    fig.add_trace(
        go.Scatter(
            x=risk_scores,
            y=expected_returns,
            mode='markers+text',
            text=portfolio_names,
            textposition="top center",
            marker=dict(size=15, color=expected_returns, colorscale='RdYlGn'),
            name="Risk vs Return"
        ),
        row=1, col=1
    )
    
    # Risk score distribution
    fig.add_trace(
        go.Bar(x=portfolio_names, y=risk_scores, name="Risk Score"),
        row=1, col=2
    )
    
    # Max drawdown
    fig.add_trace(
        go.Bar(x=portfolio_names, y=max_drawdowns, name="Max Drawdown %"),
        row=2, col=1
    )
    
    # Portfolio comparison
    fig.add_trace(
        go.Bar(x=portfolio_names, y=expected_returns, name="Expected Return %"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Portfolio Risk Analysis")
    return fig

# --- MAIN LAYOUT ---
app.layout = dbc.Container([
    dcc.Store(id='portfolios-store'),
    dcc.Store(id='market-regime-store'),
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    ),
    
    # Header
    create_header(),
    
    # Market Dashboard
    create_market_dashboard(),
    
    # Controls and Results
    dbc.Row([
        dbc.Col([
            create_controls_panel()
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Portfolio Results", className="mb-0"),
                    dbc.Spinner(html.Div(id="scan-status"), color="primary")
                ]),
                dbc.CardBody([
                    html.Div(id="portfolio-cards")
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    # Detailed Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Strategy Details", className="mb-0")),
                dbc.CardBody([
                    html.Div(id="strategy-table")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Risk Analysis Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Risk Analysis", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id="risk-metrics-chart")
                ])
            ])
        ], width=12)
    ]),
    
    # Modals
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Portfolio Details")),
        dbc.ModalBody(id="portfolio-details-content"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-portfolio-modal", className="ms-auto", n_clicks=0)
        )
    ], id="portfolio-modal", is_open=False, size="xl")
    
], fluid=True)

# --- CALLBACKS ---
@app.callback(
    [Output('market-regime', 'children'),
     Output('regime-confidence', 'children'),
     Output('vix-level', 'children'),
     Output('vix-percentile', 'children'),
     Output('vix-term-structure', 'children'),
     Output('trend-strength', 'children'),
     Output('correlation-regime', 'children'),
     Output('market-regime-store', 'data')],
    [Input('refresh-market-btn', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_market_dashboard(n_clicks, n_intervals):
    """Update market dashboard"""
    data = fetch_api("/market-regime")
    if not data:
        return "Error", "", "Error", "", "Error", "Error", "", {}
    
    regime = data.get('regime', 'Unknown')
    vix_level = data.get('vix_level', 0)
    vix_percentile = data.get('volatility_percentile', 0)
    term_structure = data.get('vix_term_structure', 'Unknown')
    trend_strength = data.get('trend_strength', 0)
    correlation_regime = data.get('correlation_regime', 'Unknown')
    confidence = data.get('confidence', 0)
    
    return (
        regime,
        f"Confidence: {confidence:.1%}",
        f"{vix_level:.2f}",
        f"{vix_percentile:.1f}th percentile",
        term_structure,
        f"{trend_strength:.1f}%",
        correlation_regime,
        data
    )

@app.callback(
    [Output('portfolios-store', 'data'),
     Output('portfolio-cards', 'children'),
     Output('strategy-table', 'children'),
     Output('risk-metrics-chart', 'figure'),
     Output('scan-status', 'children')],
    [Input('run-scan-btn', 'n_clicks')],
    [State('max-risk-slider', 'value'),
     State('min-return-slider', 'value'),
     State('strategy-types', 'value'),
     State('max-positions', 'value')]
)
def run_scan(n_clicks, max_risk, min_return, strategy_types, max_positions):
    """Run advanced options scan"""
    if n_clicks is None:
        return [], [], "No data", go.Figure(), ""
    
    # Show loading status
    loading_status = dbc.Alert("Running advanced scan... This may take a few minutes.", color="info")
    
    # Prepare scan request
    scan_data = {
        "max_risk": max_risk,
        "min_return": min_return,
        "strategy_types": strategy_types,
        "max_positions": max_positions
    }
    
    # Run scan
    result = fetch_api("/scan", method="POST", data=scan_data)
    
    if not result or 'portfolios' not in result:
        return [], [], "Scan failed", go.Figure(), dbc.Alert("Scan failed. Please try again.", color="danger")
    
    portfolios = result['portfolios']
    
    # Create portfolio cards
    portfolio_cards = create_portfolio_cards(portfolios)
    
    # Create strategy table (from first portfolio)
    all_strategies = []
    for portfolio in portfolios:
        all_strategies.extend(portfolio.get('strategies', []))
    
    strategy_table = create_strategy_table(all_strategies)
    
    # Create risk metrics chart
    risk_chart = create_risk_metrics_chart(portfolios)
    
    return portfolios, portfolio_cards, strategy_table, risk_chart, ""

@app.callback(
    [Output('portfolio-cards', 'children'),
     Output('strategy-table', 'children'),
     Output('risk-metrics-chart', 'figure')],
    [Input('clear-results-btn', 'n_clicks')]
)
def clear_results(n_clicks):
    """Clear all results"""
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    return [], "No data", go.Figure()

@app.callback(
    [Output('portfolio-modal', 'is_open'),
     Output('portfolio-details-content', 'children')],
    [Input({'type': 'portfolio-details-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input('close-portfolio-modal', 'n_clicks')],
    [State('portfolios-store', 'data')]
)
def toggle_portfolio_modal(portfolio_clicks, close_clicks, portfolios_data):
    """Toggle portfolio details modal"""
    ctx = callback_context
    if not ctx.triggered:
        return False, ""
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    if 'close-portfolio-modal' in trigger_id:
        return False, ""
    
    if 'portfolio-details-btn' in trigger_id:
        # Extract portfolio index
        portfolio_index = int(trigger_id.split('"index":')[1].split('}')[0])
        
        if portfolios_data and portfolio_index < len(portfolios_data):
            portfolio = portfolios_data[portfolio_index]
            
            # Create detailed content
            content = dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H5("Portfolio Metrics"),
                        html.P(f"Total Strategies: {len(portfolio.get('strategies', []))}"),
                        html.P(f"Expected Return: {format_percentage(portfolio.get('expected_return', 0) * 100)}"),
                        html.P(f"Risk Score: {portfolio.get('risk_score', 0):.2f}"),
                        html.P(f"Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}"),
                    ], width=6),
                    dbc.Col([
                        html.H5("Greeks Summary"),
                        html.P(f"Total Delta: {portfolio.get('total_delta', 0):.2f}"),
                        html.P(f"Total Gamma: {portfolio.get('total_gamma', 0):.2f}"),
                        html.P(f"Total Theta: {portfolio.get('total_theta', 0):.2f}"),
                        html.P(f"Total Vega: {portfolio.get('total_vega', 0):.2f}"),
                    ], width=6)
                ]),
                html.Hr(),
                html.H5("Strategy Details"),
                create_strategy_table(portfolio.get('strategies', []))
            ])
            
            return True, content
    
    return False, ""

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)