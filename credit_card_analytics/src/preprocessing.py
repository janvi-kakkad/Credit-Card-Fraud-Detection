"""
preprocessing.py — Data Loading, Cleaning, and Quality Reporting
=================================================================

Handles all ETL operations on the raw CC_GENERAL.csv:
  • NULL imputation (median for MINIMUM_PAYMENTS, drop for CREDIT_LIMIT)
  • Duplicate removal
  • Data-type enforcement (float64 for all financial columns)
  • Data-quality report generation (shape, nulls, descriptive stats, skewness)
  • EDA plot generation (histograms, heatmap, boxplots, scatter)

All file paths are sourced from ``src.utils`` — nothing is hard-coded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    CLEANED_CSV_PATH,
    FINANCIAL_COLUMNS,
    PLOT_DPI,
    PLOTS_DIR,
    RAW_CSV_PATH,
    ensure_directories,
    progress,
)


# ──────────────────────────────────────────────────────────────────
# 1. Loading
# ──────────────────────────────────────────────────────────────────
def load_raw_data(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """
    Load the raw credit-card dataset from CSV.

    Parameters
    ----------
    path : Path
        Absolute path to CC_GENERAL.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with CUST_ID preserved as the index.
    """
    dataframe = pd.read_csv(path)
    dataframe.set_index("CUST_ID", inplace=True)
    progress(f"Raw data loaded: {dataframe.shape[0]} rows, "
             f"{dataframe.shape[1]} columns")
    return dataframe


# ──────────────────────────────────────────────────────────────────
# 2. Cleaning
# ──────────────────────────────────────────────────────────────────
def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to the raw dataframe.

    Steps
    -----
    1. Impute MINIMUM_PAYMENTS nulls with column median.
    2. Drop rows where CREDIT_LIMIT is null (~1 row).
    3. Remove exact duplicate rows.
    4. Cast every financial column to float64.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw dataframe (index = CUST_ID).

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    shape_before = dataframe.shape
    nulls_before = dataframe.isnull().sum()

    # Median imputation for MINIMUM_PAYMENTS (right-skewed)
    median_min_payments = dataframe["MINIMUM_PAYMENTS"].median()
    dataframe["MINIMUM_PAYMENTS"].fillna(median_min_payments, inplace=True)
    progress(f"MINIMUM_PAYMENTS nulls imputed with median "
             f"(₹{median_min_payments:,.2f})")

    # Drop rows with null CREDIT_LIMIT (negligible loss)
    credit_limit_nulls = dataframe["CREDIT_LIMIT"].isnull().sum()
    dataframe.dropna(subset=["CREDIT_LIMIT"], inplace=True)
    progress(f"Dropped {credit_limit_nulls} row(s) with null CREDIT_LIMIT")

    # Remove exact duplicates
    duplicates_count = dataframe.duplicated().sum()
    dataframe.drop_duplicates(inplace=True)
    progress(f"Removed {duplicates_count} exact duplicate row(s)")

    # Enforce float64 for all financial columns
    for column in FINANCIAL_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype(np.float64)
    progress("All financial columns cast to float64")

    nulls_after = dataframe.isnull().sum()
    progress(f"Cleaning complete: {shape_before[0]} → "
             f"{dataframe.shape[0]} rows retained")

    return dataframe, nulls_before, nulls_after, shape_before


def print_quality_report(
    dataframe: pd.DataFrame,
    nulls_before: pd.Series,
    nulls_after: pd.Series,
    shape_before: tuple,
) -> None:
    """
    Print a comprehensive data-quality report to the console.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe.
    nulls_before : pd.Series
        Null counts per column before cleaning.
    nulls_after : pd.Series
        Null counts per column after cleaning.
    shape_before : tuple
        (rows, cols) of the raw dataframe.
    """
    print("\n" + "=" * 70)
    print("  DATA QUALITY REPORT")
    print("=" * 70)

    print(f"\n  Shape before cleaning : {shape_before}")
    print(f"  Shape after cleaning  : {dataframe.shape}")

    print("\n  ── Null Counts ──")
    null_comparison = pd.DataFrame({
        "Before": nulls_before,
        "After": nulls_after,
    })
    print(null_comparison.to_string(index=True))

    print("\n  ── Descriptive Statistics ──")
    stats = dataframe.describe().T
    stats["skewness"] = dataframe.skew()
    print(stats[["mean", "std", "min", "max", "skewness"]]
          .round(2).to_string())
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────
# 3. Persistence
# ──────────────────────────────────────────────────────────────────
def save_cleaned_data(
    dataframe: pd.DataFrame,
    path: Path = CLEANED_CSV_PATH,
) -> None:
    """
    Write the cleaned dataframe to CSV, preserving CUST_ID index.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe.
    path : Path
        Destination file path.
    """
    ensure_directories()
    dataframe.to_csv(path, index=True)
    progress(f"Cleaned data saved → {path.relative_to(path.parent.parent.parent)}")


# ──────────────────────────────────────────────────────────────────
# 4. EDA Plots
# ──────────────────────────────────────────────────────────────────
def plot_distributions(dataframe: pd.DataFrame) -> None:
    """
    Plot distribution histograms for all 17 numeric features.

    Layout: 4 × 5 grid, figure size 20 × 16, saved at PLOT_DPI.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe with numeric columns.
    """
    ensure_directories()
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for index, column in enumerate(numeric_columns):
        axes[index].hist(
            dataframe[column], bins=40, color="#4361ee",
            edgecolor="white", alpha=0.85,
        )
        axes[index].set_title(column, fontsize=10, fontweight="bold")
        axes[index].tick_params(labelsize=8)

    # Hide unused subplot slots
    for index in range(len(numeric_columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle(
        "Distribution of All Numeric Features",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = PLOTS_DIR / "01_feature_distributions.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Distribution histograms saved → {save_path.name}")


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> None:
    """
    Plot a lower-triangle correlation heatmap with annotations.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe with numeric columns.
    """
    ensure_directories()
    correlation_matrix = dataframe.select_dtypes(
        include=[np.number]
    ).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    fig, axis = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=axis,
        annot_kws={"size": 7},
    )
    axis.set_title(
        "Feature Correlation Matrix (Lower Triangle)",
        fontsize=14, fontweight="bold", pad=20,
    )
    plt.tight_layout()
    save_path = PLOTS_DIR / "02_correlation_heatmap.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Correlation heatmap saved → {save_path.name}")


def plot_outlier_boxplots(dataframe: pd.DataFrame) -> None:
    """
    Plot box plots for BALANCE, PURCHASES, CREDIT_LIMIT, PAYMENTS.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe.
    """
    ensure_directories()
    target_columns = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for index, column in enumerate(target_columns):
        sns.boxplot(
            y=dataframe[column], ax=axes[index],
            color="#7209b7", flierprops={"marker": "o", "alpha": 0.4},
        )
        axes[index].set_title(column, fontsize=12, fontweight="bold")
        axes[index].set_ylabel("Value (₹)", fontsize=10)

    fig.suptitle(
        "Outlier Distribution — Key Financial Features",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = PLOTS_DIR / "03_outlier_boxplots.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Outlier boxplots saved → {save_path.name}")


def plot_purchases_vs_credit_limit(dataframe: pd.DataFrame) -> None:
    """
    Scatter plot of PURCHASES vs CREDIT_LIMIT, coloured by
    PRC_FULL_PAYMENT quintile.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataframe.
    """
    ensure_directories()
    plot_dataframe = dataframe.copy()
    plot_dataframe["FULL_PAY_QUINTILE"] = pd.qcut(
        plot_dataframe["PRC_FULL_PAYMENT"].rank(method="first"),
        q=5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
    )

    fig, axis = plt.subplots(figsize=(12, 8))
    palette = ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"]
    for quintile_index, (label, group) in enumerate(
        plot_dataframe.groupby("FULL_PAY_QUINTILE", observed=True)
    ):
        axis.scatter(
            group["CREDIT_LIMIT"], group["PURCHASES"],
            alpha=0.5, s=15, label=label, color=palette[quintile_index],
        )

    axis.set_xlabel("Credit Limit (₹)", fontsize=12)
    axis.set_ylabel("Purchases (₹)", fontsize=12)
    axis.set_title(
        "Purchases vs Credit Limit — Coloured by Full-Payment Quintile",
        fontsize=14, fontweight="bold",
    )
    axis.legend(title="PRC_FULL_PAYMENT Quintile", fontsize=9)
    plt.tight_layout()
    save_path = PLOTS_DIR / "04_purchases_vs_credit_limit.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Scatter plot saved → {save_path.name}")
