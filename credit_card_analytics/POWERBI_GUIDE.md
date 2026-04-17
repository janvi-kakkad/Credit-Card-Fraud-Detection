# Power BI Dashboard Guide — Credit Card Customer Analytics

## Overview

This guide explains how to configure a Power BI dashboard using the four
CSV exports produced by the analytics pipeline. Each CSV is optimised for
a specific set of visuals without requiring additional transformation in
Power Query.

---

## Data Sources

| File | Path | Rows | Purpose |
|------|------|------|---------|
| `customers_master.csv` | `data/exports/` | ~8,949 | Customer-level detail with segments and fraud scores |
| `segment_summary.csv` | `data/exports/` | 4 | Aggregated segment metrics for KPI cards |
| `fraud_tiers.csv` | `data/exports/` | ~8,949 | Fraud tier classification for risk dashboards |
| `pca_coordinates.csv` | `data/exports/` | ~8,949 | PCA-projected coordinates for scatter visuals |

### Loading Instructions

1. Open Power BI Desktop → **Get Data** → **Text/CSV**
2. Load all four CSVs as separate tables
3. In **Model View**, create relationships:
   - `customers_master[CUST_ID]` → `fraud_tiers[CUST_ID]` (1:1)
   - `customers_master[CUST_ID]` → `pca_coordinates[CUST_ID]` (1:1)
   - `customers_master[SEGMENT_NAME]` → `segment_summary[SEGMENT_NAME]` (M:1)

---

## Recommended Visuals

### 1. Donut Chart — Segment Distribution

- **Source:** `segment_summary.csv`
- **Legend:** `SEGMENT_NAME`
- **Values:** `CUSTOMER_COUNT`
- **Data labels:** Show percentage
- **Colours:** Transactors=#06d6a0, Revolvers=#ef476f,
  Cash Advance Users=#ffd166, Dormant=#118ab2

### 2. Scatter Plot — PCA 2D Projection

- **Source:** `pca_coordinates.csv`
- **X-axis:** `PC1`
- **Y-axis:** `PC2`
- **Legend:** `SEGMENT_NAME`
- **Size:** `FRAUD_RISK_SCORE`
- **Tooltips:** Add `CUST_ID`, `FRAUD_RISK_SCORE`
- This replicates the interactive PCA scatter from the Plotly output

### 3. Clustered Bar Chart — Avg Utilisation by Segment

- **Source:** `segment_summary.csv`
- **Axis:** `SEGMENT_NAME`
- **Values:** `AVG_UTILIZATION`
- **Data labels:** ON (2 decimal places)
- Apply conditional formatting: red gradient for higher utilisation

### 4. KPI Cards (4 cards)

| KPI | Source | Measure |
|-----|--------|---------|
| Total Customers | `segment_summary` | `SUM(CUSTOMER_COUNT)` |
| Anomaly Rate % | DAX measure (below) | Custom |
| High-Risk Count | `fraud_tiers` | `COUNTROWS(FILTER(...))` |
| Avg Fraud Score | `customers_master` | `AVERAGE(FRAUD_RISK_SCORE)` |

### 5. Table — Fraud Tier Breakdown

- **Source:** `fraud_tiers.csv`
- **Columns:** `CUST_ID`, `SEGMENT_NAME`, `FRAUD_RISK_SCORE`,
  `FRAUD_TIER`, `CASH_DOMINANT_FLAG`, `OVERLIMIT_FLAG`
- **Conditional formatting:**
  - `FRAUD_TIER = "HIGH_RISK"` → Red background
  - `FRAUD_TIER = "MEDIUM_RISK"` → Amber background
  - `FRAUD_TIER = "LOW_RISK"` → Green background

### 6. Bar Chart — Anomaly Count by Segment

- **Source:** `segment_summary.csv`
- **Axis:** `SEGMENT_NAME`
- **Values:** `ANOMALY_COUNT`
- Stack with `ANOMALY_RATE_PCT` as tooltip

---

## Suggested DAX Measures

```dax
// 1. Anomaly Rate (%)
Anomaly Rate % =
    DIVIDE(
        CALCULATE(COUNTROWS(customers_master),
                  customers_master[ANOMALY_FLAG] = 1),
        COUNTROWS(customers_master),
        0
    ) * 100

// 2. Avg Fraud Score
Avg Fraud Score =
    AVERAGE(customers_master[FRAUD_RISK_SCORE])

// 3. High-Risk Customer Count
High Risk Count =
    CALCULATE(
        COUNTROWS(fraud_tiers),
        fraud_tiers[FRAUD_TIER] = "HIGH_RISK"
    )

// 4. Segment Customer Count (for card visuals)
Segment Count =
    COUNTROWS(customers_master)

// 5. Avg Utilisation by Segment
Avg Utilisation =
    AVERAGE(customers_master[UTILIZATION_RATIO])

// 6. Cash Dominant Count
Cash Dominant Count =
    CALCULATE(
        COUNTROWS(fraud_tiers),
        fraud_tiers[CASH_DOMINANT_FLAG] = 1
    )
```

---

## Recommended Slicers

Place these on every dashboard page for interactive filtering:

| Slicer | Source Column | Type |
|--------|-------------|------|
| Segment | `customers_master[SEGMENT_NAME]` | Dropdown |
| Fraud Tier | `fraud_tiers[FRAUD_TIER]` | Dropdown |
| Anomaly Flag | `customers_master[ANOMALY_FLAG]` | Toggle (0/1) |
| Tenure | `customers_master[TENURE]` | Range slider |

---

## Conditional Formatting Rules

### Fraud Risk Score (Table / Matrix)
- **Rules-based:**
  - Score ≥ 75 → Background: `#dc3545` (red), Font: white
  - Score 50–74 → Background: `#ffc107` (amber), Font: black
  - Score < 50 → Background: `#28a745` (green), Font: white

### Utilisation Ratio (Bar Chart / Table)
- **Gradient:**
  - Minimum (0.0) → `#06d6a0` (green)
  - Maximum (1.0) → `#ef476f` (red)

---

## Dashboard Layout Suggestion

```
┌─────────────────────────────────────────────────┐
│  KPI: Total    KPI: Anomaly   KPI: High    KPI: │
│  Customers     Rate %         Risk Count   Avg  │
│                                            Score│
├──────────────────────┬──────────────────────────┤
│  Donut Chart         │  PCA Scatter Plot        │
│  (Segment Dist.)     │  (PC1 vs PC2)            │
├──────────────────────┼──────────────────────────┤
│  Bar: Utilisation    │  Bar: Anomaly Count      │
│  by Segment          │  by Segment              │
├──────────────────────┴──────────────────────────┤
│  Table: Fraud Tier Breakdown (filterable)        │
│  [CUST_ID | SEGMENT | SCORE | TIER | FLAGS]     │
└─────────────────────────────────────────────────┘
Slicers: Segment | Fraud Tier | Anomaly Flag | Tenure
```

---

## Notes

- All monetary values in the dataset use **Indian Rupees (₹)**.
- Format currency columns in Power BI: Format → Custom → `₹#,##,##0.00`
- The PCA scatter performs best with **marker size mapped to
  FRAUD_RISK_SCORE** — larger markers indicate higher risk.
- Refresh data by re-running notebooks 01→05 and replacing the CSVs.
