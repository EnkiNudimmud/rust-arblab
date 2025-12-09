"""
Comprehensive Universe Definitions for Trading Strategies

Provides curated lists of symbols across:
- Large cap stocks (100+ S&P 500)
- Sector-specific stocks
- Cryptocurrencies (20+ pairs)
- ETFs (major and sector-specific)
"""

# S&P 500 - Top 100 by market cap (as of 2024)
SP500_TOP100 = [
    # Mega cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    # Large cap tech
    "CRM", "ACN", "CSCO", "INTC", "AMD", "IBM", "QCOM", "TXN", "INTU", "NOW",
    # Tech continued
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP", "ADI", "NXPI", "MRVL",
    # Finance
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "AXP",
    "BLK", "C", "USB", "PNC", "TFC", "COF", "BK", "AIG", "MET", "PRU",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY",
    "AMGN", "CVS", "MDT", "GILD", "CI", "ISRG", "REGN", "ZTS", "SYK", "BSX",
    # Consumer
    "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", "DIS", "CMCSA",
    "NFLX", "VZ", "T", "TMUS", "PM", "TGT", "SBUX", "LOW", "TJX", "EL",
    # Industrial & Energy
    "XOM", "CVX", "LIN", "RTX", "UNP", "HON", "CAT", "BA", "UPS", "DE",
    "GE", "MMM", "LMT", "ADP", "GD", "NSC", "EMR", "ETN", "SLB", "EOG"
]

# Technology Sector (30 stocks)
TECH_SECTOR = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", 
    "ORCL", "ADBE", "CRM", "ACN", "CSCO", "INTC", "AMD", "IBM",
    "QCOM", "TXN", "INTU", "NOW", "AMAT", "MU", "LRCX", "KLAC",
    "SNPS", "CDNS", "MCHP", "ADI", "NXPI", "MRVL"
]

# Financial Sector (30 stocks)
FINANCE_SECTOR = [
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "AXP",
    "BLK", "C", "USB", "PNC", "TFC", "COF", "BK", "AIG", "MET", "PRU",
    "CB", "MMC", "AON", "AJG", "TRV", "ALL", "PGR", "HIG", "AFL", "WRB"
]

# Healthcare Sector (30 stocks)
HEALTHCARE_SECTOR = [
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY",
    "AMGN", "CVS", "MDT", "GILD", "CI", "ISRG", "REGN", "ZTS", "SYK", "BSX",
    "VRTX", "HCA", "ELV", "HUM", "MCK", "COR", "IDXX", "IQV", "RMD", "DXCM"
]

# Energy Sector (20 stocks)
ENERGY_SECTOR = [
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "HES", "HAL", "BKR", "FANG", "DVN", "EQT", "MRO", "APA", "OVV"
]

# Consumer Discretionary (30 stocks)
CONSUMER_SECTOR = [
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "ABNB",
    "CMG", "MAR", "GM", "F", "DHI", "LEN", "NVR", "PHM", "ORLY", "AZO",
    "YUM", "DG", "DLTR", "ROST", "ULTA", "DPZ", "POOL", "BBY", "GPC", "KMX"
]

# Industrial Sector (30 stocks)
INDUSTRIAL_SECTOR = [
    "UNP", "HON", "CAT", "BA", "UPS", "DE", "GE", "MMM", "LMT", "RTX",
    "ADP", "GD", "NSC", "EMR", "ETN", "ITW", "PH", "TDG", "CSX", "WM",
    "FDX", "NOC", "CARR", "PCAR", "RSG", "OTIS", "ROK", "AME", "FAST", "VRSK"
]

# Cryptocurrency - Major Pairs (25 pairs)
CRYPTO_MAJOR = [
    "BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT", "BINANCE:XRPUSDT",
    "BINANCE:ADAUSDT", "BINANCE:DOGEUSDT", "BINANCE:SOLUSDT", "BINANCE:MATICUSDT",
    "BINANCE:DOTUSDT", "BINANCE:AVAXUSDT", "BINANCE:SHIBUSDT", "BINANCE:LTCUSDT",
    "BINANCE:TRXUSDT", "BINANCE:LINKUSDT", "BINANCE:ATOMUSDT", "BINANCE:XLMUSDT",
    "BINANCE:UNIUSDT", "BINANCE:ETCUSDT", "BINANCE:ALGOUSDT", "BINANCE:NEARUSDT",
    "BINANCE:APTUSDT", "BINANCE:ARBUSDT", "BINANCE:OPUSDT", "BINANCE:INJUSDT",
    "BINANCE:SUIUSDT"
]

# Cryptocurrency - DeFi Focused (15 pairs)
CRYPTO_DEFI = [
    "BINANCE:UNIUSDT", "BINANCE:AAVEUSDT", "BINANCE:MKRUSDT", "BINANCE:CRVUSDT",
    "BINANCE:COMPUSDT", "BINANCE:SNXUSDT", "BINANCE:SUSHIUSDT", "BINANCE:YFIUSDT",
    "BINANCE:1INCHUSDT", "BINANCE:LDOUSDT", "BINANCE:RUNEUSDT", "BINANCE:GMXUSDT",
    "BINANCE:PERPUSDT", "BINANCE:DYDXUSDT", "BINANCE:RPLUSDT"
]

# ETFs - Major Indices (10 ETFs)
ETF_MAJOR_INDICES = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "DIA",   # Dow Jones
    "VTI",   # Total US Market
    "VOO",   # S&P 500 (Vanguard)
    "VEA",   # Developed Markets ex-US
    "VWO",   # Emerging Markets
    "AGG",   # US Aggregate Bonds
    "GLD"    # Gold
]

# ETFs - Sector Specific (20 ETFs)
ETF_SECTOR = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Healthcare
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLB",   # Materials
    "XLC",   # Communication Services
    "VGT",   # Vanguard Tech
    "VFH",   # Vanguard Financials
    "VHT",   # Vanguard Healthcare
    "VDE",   # Vanguard Energy
    "VIS",   # Vanguard Industrials
    "VDC",   # Vanguard Consumer Staples
    "VCR",   # Vanguard Consumer Discretionary
    "IBB",   # Biotech
    "SOXX"   # Semiconductors
]

# ETFs - Thematic (15 ETFs)
ETF_THEMATIC = [
    "ARKK",  # ARK Innovation
    "ARKQ",  # ARK Autonomous Tech
    "ARKG",  # ARK Genomics
    "ARKW",  # ARK Next Gen Internet
    "ARKF",  # ARK Fintech
    "ICLN",  # Clean Energy
    "TAN",   # Solar Energy
    "LIT",   # Lithium & Battery
    "ROBO",  # Robotics & AI
    "HACK",  # Cybersecurity
    "CLOU",  # Cloud Computing
    "FINX",  # Fintech
    "BLOK",  # Blockchain
    "DRIV",  # Autonomous Vehicles
    "ESPO"   # Esports & Gaming
]


def get_universe(universe_name: str) -> list:
    """
    Get list of symbols for a given universe.
    
    Args:
        universe_name: Name of the universe
        
    Available universes:
        - 'sp500_top100': Top 100 S&P 500 stocks
        - 'tech': Technology sector (30)
        - 'finance': Financial sector (30)
        - 'healthcare': Healthcare sector (30)
        - 'energy': Energy sector (20)
        - 'consumer': Consumer discretionary (30)
        - 'industrial': Industrial sector (30)
        - 'crypto_major': Major cryptocurrencies (25)
        - 'crypto_defi': DeFi cryptocurrencies (15)
        - 'etf_indices': Major index ETFs (10)
        - 'etf_sector': Sector ETFs (20)
        - 'etf_thematic': Thematic ETFs (15)
        - 'all_sectors': All sector stocks (170)
        - 'all_crypto': All crypto (40)
        - 'all_etf': All ETFs (45)
        
    Returns:
        List of symbols
    """
    universes = {
        # Individual universes
        'sp500_top100': SP500_TOP100,
        'tech': TECH_SECTOR,
        'finance': FINANCE_SECTOR,
        'healthcare': HEALTHCARE_SECTOR,
        'energy': ENERGY_SECTOR,
        'consumer': CONSUMER_SECTOR,
        'industrial': INDUSTRIAL_SECTOR,
        'crypto_major': CRYPTO_MAJOR,
        'crypto_defi': CRYPTO_DEFI,
        'etf_indices': ETF_MAJOR_INDICES,
        'etf_sector': ETF_SECTOR,
        'etf_thematic': ETF_THEMATIC,
        
        # Combined universes
        'all_sectors': TECH_SECTOR + FINANCE_SECTOR + HEALTHCARE_SECTOR + 
                       ENERGY_SECTOR + CONSUMER_SECTOR + INDUSTRIAL_SECTOR,
        'all_crypto': CRYPTO_MAJOR + CRYPTO_DEFI,
        'all_etf': ETF_MAJOR_INDICES + ETF_SECTOR + ETF_THEMATIC,
        
        # Legacy aliases
        'sp500_tech': TECH_SECTOR,
        'crypto_top': CRYPTO_MAJOR[:10]
    }
    
    return universes.get(universe_name, TECH_SECTOR)


def get_available_universes() -> dict:
    """
    Get dictionary of all available universes with descriptions and sizes.
    
    Returns:
        Dict with universe names, descriptions, and sizes
    """
    return {
        'sp500_top100': {'desc': 'Top 100 S&P 500 stocks by market cap', 'size': len(SP500_TOP100)},
        'tech': {'desc': 'Technology sector stocks', 'size': len(TECH_SECTOR)},
        'finance': {'desc': 'Financial sector stocks', 'size': len(FINANCE_SECTOR)},
        'healthcare': {'desc': 'Healthcare sector stocks', 'size': len(HEALTHCARE_SECTOR)},
        'energy': {'desc': 'Energy sector stocks', 'size': len(ENERGY_SECTOR)},
        'consumer': {'desc': 'Consumer discretionary stocks', 'size': len(CONSUMER_SECTOR)},
        'industrial': {'desc': 'Industrial sector stocks', 'size': len(INDUSTRIAL_SECTOR)},
        'crypto_major': {'desc': 'Major cryptocurrencies', 'size': len(CRYPTO_MAJOR)},
        'crypto_defi': {'desc': 'DeFi-focused cryptocurrencies', 'size': len(CRYPTO_DEFI)},
        'etf_indices': {'desc': 'Major index ETFs', 'size': len(ETF_MAJOR_INDICES)},
        'etf_sector': {'desc': 'Sector-specific ETFs', 'size': len(ETF_SECTOR)},
        'etf_thematic': {'desc': 'Thematic/specialty ETFs', 'size': len(ETF_THEMATIC)},
        'all_sectors': {'desc': 'All sector stocks combined', 'size': len(TECH_SECTOR + FINANCE_SECTOR + HEALTHCARE_SECTOR + ENERGY_SECTOR + CONSUMER_SECTOR + INDUSTRIAL_SECTOR)},
        'all_crypto': {'desc': 'All crypto pairs combined', 'size': len(CRYPTO_MAJOR + CRYPTO_DEFI)},
        'all_etf': {'desc': 'All ETFs combined', 'size': len(ETF_MAJOR_INDICES + ETF_SECTOR + ETF_THEMATIC)},
    }


if __name__ == "__main__":
    print("Available Universes:")
    print("=" * 60)
    
    for name, info in get_available_universes().items():
        print(f"{name:20} {info['size']:3} symbols - {info['desc']}")
    
    print("\n" + "=" * 60)
    print(f"\nExample - Tech Sector ({len(TECH_SECTOR)} stocks):")
    print(TECH_SECTOR[:10], "...")
    
    print(f"\nExample - Crypto Major ({len(CRYPTO_MAJOR)} pairs):")
    print(CRYPTO_MAJOR[:10], "...")
