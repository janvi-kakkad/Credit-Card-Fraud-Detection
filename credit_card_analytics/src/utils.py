"""
utils.py — Shared Constants, Path Helpers, and Formatting Utilities
====================================================================

Centralises every path reference and reusable helper so that no
other module in the project contains hard-coded directory strings.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Project root is one level above `src/`
# ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────
# Data directories
# ──────────────────────────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"

# ──────────────────────────────────────────────────────────────────
# Output directories
# ──────────────────────────────────────────────────────────────────
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ──────────────────────────────────────────────────────────────────
# Canonical file paths
# ──────────────────────────────────────────────────────────────────
RAW_CSV_PATH = RAW_DATA_DIR / "CC_GENERAL.csv"
CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_customers.csv"
FEATURED_CSV_PATH = PROCESSED_DATA_DIR / "featured_customers.csv"
MASTER_CSV_PATH = PROCESSED_DATA_DIR / "master_customers.csv"
SCALER_PATH = PROCESSED_DATA_DIR / "standard_scaler.joblib"

# ──────────────────────────────────────────────────────────────────
# Feature lists (single source of truth)
# ──────────────────────────────────────────────────────────────────
FINANCIAL_COLUMNS = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT",
    "TENURE",
]

ENGINEERED_FEATURES = [
    "UTILIZATION_RATIO",
    "PAYMENT_TO_BALANCE",
    "SPEND_VELOCITY",
    "INSTALLMENT_FREQUENCY",
    "CASH_ADVANCE_RATIO",
    "PURCHASE_TYPE_RATIO",
]

CLUSTERING_FEATURES = [
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
    "UTILIZATION_RATIO", "PAYMENT_TO_BALANCE", "SPEND_VELOCITY",
    "INSTALLMENT_FREQUENCY", "CASH_ADVANCE_RATIO",
]

# ──────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
PLOT_DPI = 150

# ──────────────────────────────────────────────────────────────────
# Currency formatting (Indian norms)
# ──────────────────────────────────────────────────────────────────
CURRENCY_SYMBOL = "₹"


def format_inr(value: float) -> str:
    """
    Format a numeric value using the Indian numbering system.

    Parameters
    ----------
    value : float
        The monetary amount to format.

    Returns
    -------
    str
        Formatted string with ₹ symbol and Indian comma grouping.
        Example: ₹1,23,456.78
    """
    is_negative = value < 0
    value = abs(value)
    integer_part = int(value)
    decimal_part = f"{value - integer_part:.2f}"[1:]  # ".xx"

    # Indian grouping: last 3 digits, then groups of 2
    s = str(integer_part)
    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        remaining = s[:-3]
        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        formatted = ",".join(groups) + "," + last_three

    sign = "-" if is_negative else ""
    return f"{sign}{CURRENCY_SYMBOL}{formatted}{decimal_part}"


def ensure_directories() -> None:
    """
    Create every project directory if it does not already exist.

    Called at the top of each notebook to guarantee the folder
    tree is present before any file I/O.
    """
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXPORTS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def progress(message: str) -> None:
    """
    Print a timestamped progress message to the console.

    Parameters
    ----------
    message : str
        Human-readable status update (e.g., "Cleaning complete").
    """
    print(f"  ✓ {message}")
