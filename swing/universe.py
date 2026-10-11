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

# Mid-sized companies (mostly S&P MidCap 400 / smaller S&P 500 names today).
# The same survivorship caveat applies, more strongly: mid caps that shrank or
# went bust since 2010 are missing.
MID_CAPS = [
    "DECK", "WSM", "EME", "CSL", "RS", "LII", "WSO", "SAIA", "FIX", "TOL",
    "PHM", "NVR", "OC", "BLD", "TTC", "GGG", "NDSN", "IDXX", "MTD", "WAT",
    "TECH", "CRL", "WST", "POOL", "ODFL", "JBHT", "CHRW", "EXPD", "LSTR", "KNX",
    "AOS", "MAS", "LEA", "BWA", "GNTX", "THO", "PII", "BC", "CROX", "SKX",
    "LULU", "ULTA", "DKS", "RH", "TSCO", "CASY", "DPZ", "TXRH", "CMG", "DRI",
    "ROST", "CPRT", "FICO", "MANH", "TYL", "JKHY", "EPAM", "FDS", "MSCI", "MKTX",
    "CBOE", "NDAQ", "SEIC", "RJF", "EWBC", "CFR", "ZION", "FITB", "HBAN", "MTB",
    "STLD", "NUE", "CLF", "ATI", "CRS", "MLM", "VMC", "EXP", "AYI", "HUBB",
    "ITT", "IEX", "DOV", "ROK", "AME", "TDY", "HEI", "TXT", "CW", "WWD",
    "ALK", "LUV", "UAL", "DAL", "MKSI", "ENTG", "TER", "MPWR", "SWKS", "ON",
    "MCHP", "NTAP", "AKAM", "FFIV", "CIEN", "ZBRA", "TRMB", "FTNT", "CDNS", "SNPS",
    "PTC", "HOLX", "ALGN", "DXCM", "RMD", "COO", "STE", "MOH", "CNC", "EXEL",
    "INCY", "NBIX", "UTHR", "REGN", "VRTX", "BMRN", "JAZZ", "HRL", "SJM", "MKC",
    "CHD", "CLX", "HSY", "INGR", "DAR",
]

UNIVERSES = {"large": LARGE_CAPS, "all": LARGE_CAPS + MID_CAPS}

MARKET = "SPY"
