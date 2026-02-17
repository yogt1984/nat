"""
Configuration for Week 1-2 Validation
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
VALIDATION_DIR = PROJECT_ROOT / "exploration" / "validation"
RESULTS_DIR = VALIDATION_DIR / "results"

# Ensure directories exist
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperliquid API
HYPERLIQUID_REST_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"

# Assets to analyze
ASSETS = ["BTC", "ETH", "SOL"]

# Feature extraction settings
EMISSION_INTERVAL_MS = 100  # Feature emission every 100ms
BOOK_LEVELS = 10
TRADE_BUFFER_SECONDS = 60

# Validation settings
MIN_SAMPLES_FOR_VALIDATION = 5000  # ~8 minutes of data at 100ms
MI_SIGNIFICANCE_THRESHOLD = 0.05   # Minimum MI to consider signal exists
REGIME_THRESHOLD_LOW = 0.35        # Below this = low entropy (trend regime)
REGIME_THRESHOLD_HIGH = 0.65       # Above this = high entropy (mean-reversion)

# Backtesting settings
INITIAL_CAPITAL = 10000.0
MAKER_FEE = -0.0002  # Rebate
TAKER_FEE = 0.0005
POSITION_SIZE_PCT = 0.1  # 10% of capital per trade
