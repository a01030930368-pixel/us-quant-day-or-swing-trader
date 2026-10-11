"""Stocks the backtest may trade: large, mid-sized and small US companies.

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

# Small companies (mostly Russell 2000 / S&P SmallCap 600 names today), the
# kind of fast-moving stocks most swing legends traded. Survivorship cuts both
# ways here: small caps that went bust are missing (results look better), but
# small caps that grew into large caps are missing too (results look worse).
SMALL_CAPS = [
    "AAON", "ACLS", "AEIS", "AEO", "ALKS", "AMKR", "AMN", "ANF", "ARCB", "ARWR",
    "ASGN", "ATGE", "AVAV", "AVNT", "AX", "AZZ", "BCC", "BCPC", "BDC", "BJRI",
    "BKE", "BL", "BLKB", "BMI", "BOOT", "BOX", "CAKE", "CALM", "CALX", "CARG",
    "CATY", "CBRL", "CENTA", "CEVA", "CHEF", "CMC", "CNMD", "COHU", "CORT", "CPRX",
    "CRUS", "CRVL", "CSGS", "CSWI", "CVCO", "CVLT", "DIOD", "DORM", "DY", "ECPG",
    "EGHT", "ENSG", "EPAC", "ESE", "EXLS", "EXPO", "FELE", "FN", "FORM", "FSS",
    "GBX", "GES", "GIII", "GMS", "GPI", "HALO", "HCSG", "HELE", "HIBB", "HLIT",
    "HQY", "HWKN", "IART", "IDCC", "IIIN", "INSM", "IOSP", "IPAR", "IRBT", "ITGR",
    "ITRI", "JACK", "JBSS", "JOE", "KAI", "KALU", "KLIC", "KNSL", "KTOS", "LANC",
    "LGND", "LNTH", "LOPE", "LPX", "LZB", "MATX", "MED", "MGPI", "MHO", "MLI",
    "MMSI", "MOD", "MTH", "MYGN", "MYRG", "NEOG", "NOVT", "NSIT", "NX", "OFIX",
    "OII", "OSIS", "OXM", "PATK", "PDFS", "PLAB", "PLUS", "PLXS", "POWI", "PRGS",
    "PRIM", "QLYS", "RAMP", "RDNT", "RGEN", "ROCK", "SANM", "SCSC", "SHOO", "SIG",
    "SKYW", "SLAB", "SMTC", "SPSC", "SSTK", "STAA", "STRL", "SUPN", "SXI", "TBBK",
    "TNC", "TREX", "TTEC", "TTMI", "UCTT", "UFPI", "UFPT", "USPH", "VECO", "VICR",
    "VRTS", "WDFC", "WGO", "WING", "WIRE", "WOR", "YELP", "ZUMZ",
]

UNIVERSES = {"large": LARGE_CAPS, "all": LARGE_CAPS + MID_CAPS,
             "small": SMALL_CAPS, "every": LARGE_CAPS + MID_CAPS + SMALL_CAPS}
UNIVERSE_LABELS = {"large": "미국 대형주", "all": "미국 대형주 + 중형주",
                   "small": "미국 소형주", "every": "미국 대형주 + 중형주 + 소형주"}

MARKET = "SPY"
