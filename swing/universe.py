"""Stocks the backtest may trade: about 100 large, liquid US companies.

These are today's big names, so the list quietly leaves out companies that
shrank or disappeared since 2010 (survivorship bias). Results look somewhat
better than a real investor would have seen; treat them as an upper bound.
"""

LARGE_CAPS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "UNH", "HD", "PG", "MA", "BAC", "XOM", "CVX", "ABBV", "KO",
    "PEP", "COST", "WMT", "MRK", "PFE", "LLY", "AVGO", "ADBE", "CSCO", "ORCL",
    "CRM", "ACN", "TMO", "ABT", "DHR", "MCD", "NKE", "DIS", "NFLX", "INTC",
    "AMD", "QCOM", "TXN", "IBM", "AMGN", "GILD", "BMY", "CAT", "DE", "HON",
    "GE", "MMM", "UPS", "UNP", "LMT", "RTX", "BA", "GS", "MS", "C",
    "WFC", "AXP", "BLK", "SCHW", "SPGI", "USB", "PNC", "LOW", "SBUX", "TGT",
    "BKNG", "CMCSA", "T", "VZ", "NEE", "DUK", "SO", "LIN", "APD", "MDT",
    "ISRG", "SYK", "BDX", "CVS", "CI", "ELV", "MO", "PM", "CL", "MDLZ",
    "KMB", "GD", "NOC", "CSX", "NSC", "FDX", "COP", "SLB", "EOG", "ADP",
    "INTU", "AMAT", "MU", "LRCX", "KLAC",
]

MARKET = "SPY"
