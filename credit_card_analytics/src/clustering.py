"""
clustering.py — StandardScaler, K-Means (k=4), and PCA Visualisation
=====================================================================

Orchestrates the full clustering workflow:
  1. Feature normalisation via StandardScaler (persisted with joblib).
  2. Elbow + Silhouette analysis for optimal k.
  3. K-Means fitting (k=4, n_init=20, max_iter=500).
  4. Centroid-driven persona mapping (Transactors, Revolvers, etc.).
  5. PCA 2-D scatter with labelled centroids (static + interactive).
"""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import (
    CLUSTERING_FEATURES,
    PLOT_DPI,
    PLOTS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SCALER_PATH,
    ensure_directories,
    format_inr,
    progress,
)

# Try importing kneed for programmatic elbow detection
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# 1. Normalisation
# ──────────────────────────────────────────────────────────────────
def normalise_features(
    dataframe: pd.DataFrame,
    features: list = CLUSTERING_FEATURES,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to the selected clustering features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature-enriched dataframe.
    features : list
        Column names to scale.

    Returns
    -------
    scaled_array : np.ndarray
        Scaled feature matrix.
    scaler : StandardScaler
        Fitted scaler (persisted to disk for reuse).
    """
    ensure_directories()
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(dataframe[features])
    joblib.dump(scaler, SCALER_PATH)
    progress(f"StandardScaler fitted and saved → {SCALER_PATH.name}")
    return scaled_array, scaler


# ──────────────────────────────────────────────────────────────────
# 2. Elbow + Silhouette Analysis
# ──────────────────────────────────────────────────────────────────
def plot_elbow_and_silhouette(
    scaled_array: np.ndarray,
    k_range: range = range(2, 13),
    silhouette_k_range: range = range(2, 11),
) -> int:
    """
    Plot inertia (elbow) and silhouette-score curves.

    Annotates the elbow point using the ``kneed`` library when
    available, otherwise defaults to k=4.

    Parameters
    ----------
    scaled_array : np.ndarray
        Normalised feature matrix.
    k_range : range
        Range of k values for the elbow plot.
    silhouette_k_range : range
        Range of k values for silhouette analysis.

    Returns
    -------
    int
        Optimal number of clusters (k).
    """
    ensure_directories()

    # ── Inertia curve ──
    inertia_values = []
    for k in k_range:
        km = KMeans(
            n_clusters=k, random_state=RANDOM_STATE,
            n_init=10, max_iter=300,
        )
        km.fit(scaled_array)
        inertia_values.append(km.inertia_)

    # Detect elbow
    if KNEED_AVAILABLE:
        kneedle = KneeLocator(
            list(k_range), inertia_values,
            curve="convex", direction="decreasing",
        )
        optimal_k = kneedle.elbow if kneedle.elbow else 4
    else:
        optimal_k = 4

    # ── Silhouette curve ──
    silhouette_scores = []
    for k in silhouette_k_range:
        km = KMeans(
            n_clusters=k, random_state=RANDOM_STATE,
            n_init=10, max_iter=300,
        )
        labels = km.fit_predict(scaled_array)
        silhouette_scores.append(silhouette_score(scaled_array, labels))

    # ── Plot both curves ──
    fig, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(16, 6))

    ax_elbow.plot(
        list(k_range), inertia_values,
        marker="o", linewidth=2, color="#4361ee",
    )
    ax_elbow.axvline(
        x=optimal_k, linestyle="--", color="#ef476f", linewidth=1.5,
        label=f"Elbow at k={optimal_k}",
    )
    ax_elbow.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax_elbow.set_ylabel("Inertia (WCSS)", fontsize=12)
    ax_elbow.set_title("Elbow Method", fontsize=13, fontweight="bold")
    ax_elbow.legend(fontsize=10)

    ax_sil.plot(
        list(silhouette_k_range), silhouette_scores,
        marker="s", linewidth=2, color="#06d6a0",
    )
    ax_sil.axvline(
        x=optimal_k, linestyle="--", color="#ef476f", linewidth=1.5,
        label=f"Selected k={optimal_k}",
    )
    ax_sil.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax_sil.set_ylabel("Silhouette Score", fontsize=12)
    ax_sil.set_title("Silhouette Analysis", fontsize=13, fontweight="bold")
    ax_sil.legend(fontsize=10)

    fig.suptitle(
        "Optimal Cluster Selection — Elbow + Silhouette Analysis",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = PLOTS_DIR / "06_elbow_silhouette.png"
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"Elbow + Silhouette plot saved → {save_path.name}")
    progress(f"Optimal k detected: {optimal_k}")
    return optimal_k


# ──────────────────────────────────────────────────────────────────
# 3. K-Means Fitting
# ──────────────────────────────────────────────────────────────────
def fit_kmeans(
    scaled_array: np.ndarray,
    n_clusters: int = 4,
) -> KMeans:
    """
    Fit K-Means with production-grade hyperparameters.

    Parameters
    ----------
    scaled_array : np.ndarray
        Normalised feature matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    KMeans
        Fitted K-Means estimator.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=500,
    )
    kmeans.fit(scaled_array)
    progress(f"K-Means fitted: k={n_clusters}, "
             f"inertia={kmeans.inertia_:,.0f}")
    return kmeans


# ──────────────────────────────────────────────────────────────────
# 4. Persona Mapping (centroid-driven)
# ──────────────────────────────────────────────────────────────────
def assign_personas(
    dataframe: pd.DataFrame,
    kmeans: KMeans,
    features: list = CLUSTERING_FEATURES,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Map cluster integer labels to descriptive persona names
    by inspecting centroid characteristics.

    Mapping Logic
    -------------
    - Highest mean PURCHASES + PAYMENTS, lowest CASH_ADVANCE
      → **Transactors**
    - Highest mean BALANCE + UTILIZATION_RATIO, lowest PAYMENTS
      → **Revolvers**
    - Highest mean CASH_ADVANCE, lowest PURCHASES
      → **Cash Advance Users**
    - Lowest overall activity (bottom on most features)
      → **Dormant/Low Engagement**

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature-enriched dataframe.
    kmeans : KMeans
        Fitted K-Means estimator.
    features : list
        Feature names corresponding to centroid columns.

    Returns
    -------
    dataframe : pd.DataFrame
        With CLUSTER_ID and SEGMENT_NAME columns added.
    persona_map : dict
        {cluster_int: persona_string} mapping.
    """
    dataframe["CLUSTER_ID"] = kmeans.labels_

    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n── Cluster Centroids (Scaled) ──")
    print(centroids.round(3).to_string())

    # Analytical ranking to assign personas
    persona_map = {}
    assigned = set()

    # Transactors: highest purchases + payments, low cash_advance
    transactor_score = (
        centroids["PURCHASES"] + centroids["PAYMENTS"]
        - centroids["CASH_ADVANCE"]
    )
    transactor_id = _best_unassigned(transactor_score, assigned, highest=True)
    persona_map[transactor_id] = "Transactors"
    assigned.add(transactor_id)

    # Cash Advance Users: highest cash_advance, low purchases
    cash_score = (
        centroids["CASH_ADVANCE"] + centroids["CASH_ADVANCE_RATIO"]
        - centroids["PURCHASES"]
    )
    cash_id = _best_unassigned(cash_score, assigned, highest=True)
    persona_map[cash_id] = "Cash Advance Users"
    assigned.add(cash_id)

    # Revolvers: highest balance + utilization, low payment_to_balance
    revolver_score = (
        centroids["BALANCE"] + centroids["UTILIZATION_RATIO"]
        - centroids["PAYMENT_TO_BALANCE"]
    )
    revolver_id = _best_unassigned(revolver_score, assigned, highest=True)
    persona_map[revolver_id] = "Revolvers"
    assigned.add(revolver_id)

    # Dormant: whatever remains
    remaining = set(range(len(centroids))) - assigned
    for cluster_id in remaining:
        persona_map[cluster_id] = "Dormant/Low Engagement"
    assigned.update(remaining)

    # Apply mapping
    dataframe["SEGMENT_NAME"] = dataframe["CLUSTER_ID"].map(persona_map)

    print("\n── Persona Assignment ──")
    for cluster_id, name in sorted(persona_map.items()):
        count = (dataframe["CLUSTER_ID"] == cluster_id).sum()
        pct = count / len(dataframe) * 100
        print(f"  Cluster {cluster_id} → {name} "
              f"({count:,} customers, {pct:.1f}%)")

    progress("Persona labels assigned to all customers")
    return dataframe, persona_map


def _best_unassigned(
    scores: pd.Series,
    already_assigned: set,
    highest: bool = True,
) -> int:
    """
    Return the index of the best-scoring unassigned cluster.

    Parameters
    ----------
    scores : pd.Series
        Score per cluster (index = cluster id).
    already_assigned : set
        Cluster IDs that have been assigned already.
    highest : bool
        If True, pick the highest scorer; else the lowest.

    Returns
    -------
    int
        Selected cluster ID.
    """
    masked = scores.copy()
    for idx in already_assigned:
        masked.iloc[idx] = -np.inf if highest else np.inf
    return int(masked.idxmax() if highest else masked.idxmin())


# ──────────────────────────────────────────────────────────────────
# 5. Cluster Profile Report
# ──────────────────────────────────────────────────────────────────
def generate_cluster_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and print a segment-level profile report.

    Columns: count, %, mean BALANCE, mean PURCHASES, mean CREDIT_LIMIT,
    mean UTILIZATION_RATIO, mean PAYMENT_TO_BALANCE, mean CASH_ADVANCE_RATIO.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with SEGMENT_NAME column.

    Returns
    -------
    pd.DataFrame
        Profile summary (also saved as CSV).
    """
    ensure_directories()

    report_columns = {
        "BALANCE": "Avg Balance",
        "PURCHASES": "Avg Purchases",
        "CREDIT_LIMIT": "Avg Credit Limit",
        "UTILIZATION_RATIO": "Avg Utilization",
        "PAYMENT_TO_BALANCE": "Avg Pmt/Bal",
        "CASH_ADVANCE_RATIO": "Avg CA Ratio",
    }

    profile = dataframe.groupby("SEGMENT_NAME").agg(
        Count=("BALANCE", "size"),
        **{
            display_name: (column, "mean")
            for column, display_name in report_columns.items()
        },
    )
    profile["% of Total"] = (profile["Count"] / len(dataframe) * 100).round(1)

    # Reorder columns
    column_order = ["Count", "% of Total"] + list(report_columns.values())
    profile = profile[column_order]

    print("\n" + "=" * 90)
    print("  CLUSTER PROFILE REPORT")
    print("=" * 90)
    print(profile.round(3).to_string())
    print("=" * 90 + "\n")

    save_path = REPORTS_DIR / "cluster_profile_report.csv"
    profile.to_csv(save_path)
    progress(f"Cluster profile report saved → {save_path.name}")
    return profile


# ──────────────────────────────────────────────────────────────────
# 6. PCA Visualisation
# ──────────────────────────────────────────────────────────────────
def pca_visualisation(
    dataframe: pd.DataFrame,
    scaled_array: np.ndarray,
    kmeans: KMeans,
) -> pd.DataFrame:
    """
    Compute PCA (2-D and 3-D) and generate scatter plots.

    Outputs
    -------
    - Static matplotlib PNG.
    - Interactive Plotly HTML.
    - Prints explained variance for 2-component and 3-component PCA.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with SEGMENT_NAME column.
    scaled_array : np.ndarray
        Normalised feature matrix.
    kmeans : KMeans
        Fitted K-Means estimator.

    Returns
    -------
    pd.DataFrame
        Dataframe with PC1 and PC2 columns appended.
    """
    ensure_directories()

    # ── 2-component PCA ──
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    components_2d = pca_2d.fit_transform(scaled_array)
    var_2d = pca_2d.explained_variance_ratio_

    # ── 3-component PCA (for variance analysis only) ──
    pca_3d = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_3d.fit(scaled_array)
    var_3d = pca_3d.explained_variance_ratio_

    print(f"\n  PCA — 2-component explained variance: "
          f"{var_2d[0]:.1%}, {var_2d[1]:.1%} "
          f"(cumulative: {sum(var_2d):.1%})")
    print(f"  PCA — 3-component explained variance: "
          f"{var_3d[0]:.1%}, {var_3d[1]:.1%}, {var_3d[2]:.1%} "
          f"(cumulative: {sum(var_3d):.1%})")

    dataframe["PC1"] = components_2d[:, 0]
    dataframe["PC2"] = components_2d[:, 1]

    # Cluster centroids in PCA space
    centroid_pca = pca_2d.transform(kmeans.cluster_centers_)

    # ── Static matplotlib ──
    segment_colors = {
        "Transactors": "#06d6a0",
        "Revolvers": "#ef476f",
        "Cash Advance Users": "#ffd166",
        "Dormant/Low Engagement": "#118ab2",
    }

    fig, axis = plt.subplots(figsize=(14, 10))
    for segment, colour in segment_colors.items():
        mask = dataframe["SEGMENT_NAME"] == segment
        axis.scatter(
            dataframe.loc[mask, "PC1"],
            dataframe.loc[mask, "PC2"],
            c=colour, label=segment, alpha=0.45, s=15, edgecolors="none",
        )
    # Plot centroids
    for idx, (cx, cy) in enumerate(centroid_pca):
        segment_label = dataframe.loc[
            dataframe["CLUSTER_ID"] == idx, "SEGMENT_NAME"
        ].iloc[0]
        axis.scatter(
            cx, cy,
            marker="*", s=400, c="black", edgecolors="white",
            linewidths=1.5, zorder=5,
        )
        axis.annotate(
            segment_label, (cx, cy),
            textcoords="offset points", xytext=(10, 10),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    axis.set_xlabel(
        f"PC1 ({var_2d[0]:.1%} variance)", fontsize=12,
    )
    axis.set_ylabel(
        f"PC2 ({var_2d[1]:.1%} variance)", fontsize=12,
    )
    axis.set_title(
        "Customer Segments — PCA 2D Projection",
        fontsize=15, fontweight="bold",
    )
    axis.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    static_path = PLOTS_DIR / "07_pca_2d_scatter.png"
    fig.savefig(static_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    progress(f"PCA static scatter saved → {static_path.name}")

    # ── Interactive Plotly ──
    plotly_fig = px.scatter(
        dataframe.reset_index(),
        x="PC1", y="PC2",
        color="SEGMENT_NAME",
        hover_data=["CUST_ID", "BALANCE", "PURCHASES", "CREDIT_LIMIT"],
        color_discrete_map=segment_colors,
        title="Customer Segments — PCA 2D Projection (Interactive)",
        labels={
            "PC1": f"PC1 ({var_2d[0]:.1%} variance)",
            "PC2": f"PC2 ({var_2d[1]:.1%} variance)",
        },
        opacity=0.5,
    )
    plotly_fig.update_traces(marker=dict(size=5))
    plotly_fig.update_layout(
        template="plotly_dark",
        width=1000, height=700,
        legend_title_text="Customer Segment",
    )
    interactive_path = PLOTS_DIR / "07_pca_2d_interactive.html"
    plotly_fig.write_html(str(interactive_path))
    progress(f"PCA interactive scatter saved → {interactive_path.name}")

    return dataframe
