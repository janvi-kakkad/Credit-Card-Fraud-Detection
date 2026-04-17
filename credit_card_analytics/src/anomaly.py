"""
anomaly.py — Outlier Flagging and Isolation Forest Scoring
===========================================================

Two-stage anomaly detection:
  Stage 1 — IQR (1.5×) and Z-score (>3) flags on key financial columns.
  Stage 2 — Isolation Forest (contamination=0.023) on all scaled features,
            yielding a continuous FRAUD_RISK_SCORE in [0, 100].
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

from src.utils import (
    CLUSTERING_FEATURES,
    MASTER_CSV_PATH,
    PLOT_DPI,
    PLOTS_DIR,
    RANDOM_STATE,
    ensure_directories,
    progress,
)


# ──────────────────────────────────────────────────────────────────
# 1. Classic Outlier Flags (IQR + Z-Score)
# ──────────────────────────────────────────────────────────────────
def flag_classic_outliers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create boolean outlier flags using IQR 1.5× fence and Z-score > 3.

    Flags Created
    -------------
    - BALANCE_OUTLIER
    - PURCHASE_OUTLIER
    - CASH_OUTLIER
    - CLASSIC_OUTLIER_FLAG (composite: 1 if any of the three is True)

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature-enriched dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with four new boolean/integer columns.
    """
    target_columns = {
        "BALANCE": "BALANCE_OUTLIER",
        "PURCHASES": "PURCHASE_OUTLIER",
        "CASH_ADVANCE": "CASH_OUTLIER",
    }

    for source_column, flag_column in target_columns.items():
        q1 = dataframe[source_column].quantile(0.25)
        q3 = dataframe[source_column].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        z_scores = np.abs(stats.zscore(dataframe[source_column]))

        iqr_outlier = (
            (dataframe[source_column] < lower_fence)
            | (dataframe[source_column] > upper_fence)
        )
        zscore_outlier = z_scores > 3

        dataframe[flag_column] = (iqr_outlier | zscore_outlier)
        flagged_count = dataframe[flag_column].sum()
        progress(f"{flag_column}: {flagged_count} outliers "
                 f"({flagged_count / len(dataframe) * 100:.1f}%)")

    dataframe["CLASSIC_OUTLIER_FLAG"] = (
        dataframe["BALANCE_OUTLIER"]
        | dataframe["PURCHASE_OUTLIER"]
        | dataframe["CASH_OUTLIER"]
    ).astype(int)

    total_classic = dataframe["CLASSIC_OUTLIER_FLAG"].sum()
    progress(f"Composite CLASSIC_OUTLIER_FLAG: {total_classic} "
             f"({total_classic / len(dataframe) * 100:.1f}%)")
    return dataframe


# ──────────────────────────────────────────────────────────────────
# 2. Isolation Forest
# ──────────────────────────────────────────────────────────────────
def run_isolation_forest(
    dataframe: pd.DataFrame,
    scaled_array: np.ndarray,
    contamination: float = 0.023,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """
    Fit Isolation Forest and generate anomaly scores.

    New Columns
    -----------
    - ANOMALY_SCORE : raw decision_function output (lower = more anomalous).
    - ANOMALY_FLAG  : 1 if anomaly, 0 otherwise.
    - FRAUD_RISK_SCORE : ANOMALY_SCORE normalised to [0, 100]
      (100 = highest risk).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature-enriched, clustered dataframe.
    scaled_array : np.ndarray
        Normalised feature matrix (10 clustering features).
    contamination : float
        Expected proportion of anomalies (0.023 = 2.3%).
    n_estimators : int
        Number of isolation trees.

    Returns
    -------
    pd.DataFrame
        Dataframe with anomaly columns appended.
    """
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
    )
    predictions = iso_forest.fit_predict(scaled_array)
    raw_scores = iso_forest.decision_function(scaled_array)

    dataframe["ANOMALY_SCORE"] = raw_scores
    dataframe["ANOMALY_FLAG"] = (predictions == -1).astype(int)

    # Normalise to [0, 100] — lower raw score → higher risk
    score_min = raw_scores.min()
    score_max = raw_scores.max()
    dataframe["FRAUD_RISK_SCORE"] = (
        (1 - (raw_scores - score_min) / (score_max - score_min)) * 100
    ).round(2)

    total_anomalies = dataframe["ANOMALY_FLAG"].sum()
    anomaly_pct = total_anomalies / len(dataframe) * 100
    progress(f"Isolation Forest complete: {total_anomalies} anomalies "
             f"({anomaly_pct:.1f}%)")
    return dataframe


# ──────────────────────────────────────────────────────────────────
# 3. Validation & Analysis
# ──────────────────────────────────────────────────────────────────
def print_anomaly_analysis(dataframe: pd.DataFrame) -> None:
    """
    Print anomaly breakdown by segment and cross-tabulation with
    classic outlier flags.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with ANOMALY_FLAG, CLASSIC_OUTLIER_FLAG, SEGMENT_NAME.
    """
    total_anomalies = dataframe["ANOMALY_FLAG"].sum()
    anomaly_pct = total_anomalies / len(dataframe) * 100

    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION SUMMARY")
    print("=" * 70)
    print(f"  Total anomalies flagged : {total_anomalies:,}")
    print(f"  Percentage of dataset   : {anomaly_pct:.2f}%")

    print("\n  ── Anomaly Breakdown by Segment ──")
    segment_breakdown = dataframe.groupby("SEGMENT_NAME").agg(
        Total=("ANOMALY_FLAG", "size"),
        Anomalies=("ANOMALY_FLAG", "sum"),
    )
    segment_breakdown["Anomaly %"] = (
        segment_breakdown["Anomalies"]
        / segment_breakdown["Total"] * 100
    ).round(2)
    print(segment_breakdown.to_string())

    print("\n  ── Cross-Tabulation: Isolation Forest × Classic Outliers ──")
    cross_tab = pd.crosstab(
        dataframe["ANOMALY_FLAG"],
        dataframe["CLASSIC_OUTLIER_FLAG"],
        rownames=["IF Anomaly"],
        colnames=["Classic Outlier"],
    )
    print(cross_tab.to_string())

    agreement = (
        (dataframe["ANOMALY_FLAG"] == 1)
        & (dataframe["CLASSIC_OUTLIER_FLAG"] == 1)
    ).sum()
    if total_anomalies > 0:
        agreement_rate = agreement / total_anomalies * 100
        print(f"\n  Agreement rate (both flagged): "
              f"{agreement}/{total_anomalies} = {agreement_rate:.1f}%")
    print("=" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────
# 4. Anomaly Visualisations
# ──────────────────────────────────────────────────────────────────
def plot_anomaly_violin(dataframe: pd.DataFrame) -> None:
    """
    Violin plot of anomaly-score distribution per segment.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with ANOMALY_SCORE and SEGMENT_NAME.
    """
    ensure_directories()
    fig, axis = plt.subplots(figsize=(14, 7))
    sns.violinplot(
        data=dataframe,
        x="SEGMENT_NAME",
        y="ANOMALY_SCORE",
        palette={
            "Transactors": "#06d6a0",
            "Revolvers": "#ef476f",
            "Cash Advance Users": "#ffd166",
            "Dormant/Low Engagement": "#118ab2",
        },
        inner="box",
        ax=axis,
    )
    axis.set_title(
        "Anomaly Score Distribution by Segment",
        fontsize=14, fontweight="bold",
    )
    axis.set_xlabel("Customer Segment", fontsize=12)
    axis.set_ylabel("Anomaly Score (lower = more anomalous)", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_path = PLOTS_DIR / "08_anomaly_violin.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Anomaly violin plot saved → {save_path.name}")


def plot_fraud_risk_scatter(dataframe: pd.DataFrame) -> None:
    """
    Scatter of UTILIZATION_RATIO vs CASH_ADVANCE_RATIO, coloured by
    FRAUD_RISK_SCORE using a red-yellow-green colourmap.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with UTILIZATION_RATIO, CASH_ADVANCE_RATIO,
        and FRAUD_RISK_SCORE.
    """
    ensure_directories()
    fig, axis = plt.subplots(figsize=(12, 8))
    scatter = axis.scatter(
        dataframe["UTILIZATION_RATIO"],
        dataframe["CASH_ADVANCE_RATIO"],
        c=dataframe["FRAUD_RISK_SCORE"],
        cmap="RdYlGn_r",  # Red=high risk, green=low risk
        alpha=0.6,
        s=12,
        edgecolors="none",
    )
    colour_bar = plt.colorbar(scatter, ax=axis)
    colour_bar.set_label("Fraud Risk Score (0–100)", fontsize=11)

    axis.set_xlabel("Utilization Ratio", fontsize=12)
    axis.set_ylabel("Cash Advance Ratio", fontsize=12)
    axis.set_title(
        "Fraud Risk — Utilization vs Cash Advance Ratio",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    save_path = PLOTS_DIR / "09_fraud_risk_scatter.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Fraud risk scatter saved → {save_path.name}")


# ──────────────────────────────────────────────────────────────────
# 5. Persistence
# ──────────────────────────────────────────────────────────────────
def save_master_data(
    dataframe: pd.DataFrame,
    path: Path = MASTER_CSV_PATH,
) -> None:
    """
    Save the final master dataframe with all original, engineered,
    cluster, and anomaly columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Complete master dataframe.
    path : Path
        Destination file path.
    """
    ensure_directories()
    dataframe.to_csv(path, index=True)
    progress(f"Master data saved → "
             f"{path.relative_to(path.parent.parent.parent)} "
             f"({dataframe.shape[0]} rows, {dataframe.shape[1]} cols)")
