"""
features.py — Derived Feature Engineering
==========================================

Engineers exactly six behavioural ratios that capture spending
patterns, repayment discipline, and cash-advance reliance.

Each ratio is clamped or offset to avoid division-by-zero and
to keep values within interpretable bounds.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    FEATURED_CSV_PATH,
    PLOT_DPI,
    PLOTS_DIR,
    ensure_directories,
    progress,
)


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add six derived behavioural features to the dataframe.

    Features
    --------
    1. UTILIZATION_RATIO      — BALANCE / CREDIT_LIMIT, clamped [0, 1].
    2. PAYMENT_TO_BALANCE     — PAYMENTS / (BALANCE + 1).
    3. SPEND_VELOCITY         — PURCHASES / CREDIT_LIMIT, clamped [0, 1].
    4. INSTALLMENT_FREQUENCY  — INSTALLMENTS_PURCHASES / (PURCHASES + 1).
    5. CASH_ADVANCE_RATIO     — CASH_ADVANCE / (PURCHASES + CASH_ADVANCE + 1).
    6. PURCHASE_TYPE_RATIO    — ONEOFF_PURCHASES / (PURCHASES + 1).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe containing the original 17 features.

    Returns
    -------
    pd.DataFrame
        Dataframe with 6 new columns appended.
    """
    dataframe["UTILIZATION_RATIO"] = np.clip(
        dataframe["BALANCE"] / dataframe["CREDIT_LIMIT"], 0, 1,
    )
    progress("Engineered → UTILIZATION_RATIO (balance / credit_limit)")

    dataframe["PAYMENT_TO_BALANCE"] = (
        dataframe["PAYMENTS"] / (dataframe["BALANCE"] + 1)
    )
    progress("Engineered → PAYMENT_TO_BALANCE (payments / (balance + 1))")

    dataframe["SPEND_VELOCITY"] = np.clip(
        dataframe["PURCHASES"] / dataframe["CREDIT_LIMIT"], 0, 1,
    )
    progress("Engineered → SPEND_VELOCITY (purchases / credit_limit)")

    dataframe["INSTALLMENT_FREQUENCY"] = (
        dataframe["INSTALLMENTS_PURCHASES"] / (dataframe["PURCHASES"] + 1)
    )
    progress("Engineered → INSTALLMENT_FREQUENCY "
             "(installment_purchases / (purchases + 1))")

    dataframe["CASH_ADVANCE_RATIO"] = (
        dataframe["CASH_ADVANCE"]
        / (dataframe["PURCHASES"] + dataframe["CASH_ADVANCE"] + 1)
    )
    progress("Engineered → CASH_ADVANCE_RATIO "
             "(cash_advance / (purchases + cash_advance + 1))")

    dataframe["PURCHASE_TYPE_RATIO"] = (
        dataframe["ONEOFF_PURCHASES"] / (dataframe["PURCHASES"] + 1)
    )
    progress("Engineered → PURCHASE_TYPE_RATIO "
             "(oneoff_purchases / (purchases + 1))")

    progress(f"Feature engineering complete — "
             f"{dataframe.shape[1]} total columns")
    return dataframe


def plot_engineered_pairplot(dataframe: pd.DataFrame) -> None:
    """
    Generate a pair plot of the six engineered features,
    coloured by TENURE quartile.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature-enriched dataframe (must include TENURE column).
    """
    ensure_directories()
    engineered_columns = [
        "UTILIZATION_RATIO", "PAYMENT_TO_BALANCE", "SPEND_VELOCITY",
        "INSTALLMENT_FREQUENCY", "CASH_ADVANCE_RATIO", "PURCHASE_TYPE_RATIO",
    ]

    plot_dataframe = dataframe[engineered_columns + ["TENURE"]].copy()
    plot_dataframe["TENURE_QUARTILE"] = pd.qcut(
        plot_dataframe["TENURE"].rank(method="first"),
        q=4, labels=["Q1", "Q2", "Q3", "Q4"],
    )

    pair_grid = sns.pairplot(
        plot_dataframe,
        hue="TENURE_QUARTILE",
        vars=engineered_columns,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 10},
        palette="viridis",
        height=2.0,
    )
    pair_grid.figure.suptitle(
        "Engineered Features — Pair Plot (coloured by TENURE quartile)",
        y=1.02, fontsize=14, fontweight="bold",
    )
    save_path = PLOTS_DIR / "05_engineered_pairplot.png"
    pair_grid.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    progress(f"Engineered pairplot saved → {save_path.name}")


def save_featured_data(
    dataframe: pd.DataFrame,
    path: Path = FEATURED_CSV_PATH,
) -> None:
    """
    Persist the feature-enriched dataframe to CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with original + engineered features.
    path : Path
        Destination CSV path.
    """
    ensure_directories()
    dataframe.to_csv(path, index=True)
    progress(f"Featured data saved → "
             f"{path.relative_to(path.parent.parent.parent)}")
