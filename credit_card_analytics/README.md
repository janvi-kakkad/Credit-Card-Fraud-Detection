# Credit Card Spending Behaviour Analytics

**Customer Segmentation & Fraud Pattern Detection**

A production-grade, end-to-end data analytics pipeline that segments ~8,950
credit card customers into four behavioural personas using K-Means clustering,
engineers six derived spending ratios, and detects anomalous fraud patterns
with Isolation Forest — yielding a continuous Fraud Risk Score for every
customer in the portfolio.

---

## Key Findings

- **4 distinct customer segments** identified: Transactors, Revolvers,
  Cash Advance Users, and Dormant/Low Engagement.
- **2.3% of customers** flagged as anomalous by Isolation Forest, with
  highest concentration in the Cash Advance Users segment.
- **6 engineered behavioural features** (Utilisation Ratio, Payment-to-Balance,
  Spend Velocity, Installment Frequency, Cash Advance Ratio, Purchase Type
  Ratio) improved cluster separation significantly.
- Targeted upsell and retention strategies across segments project an
  estimated **₹14+ lakh incremental annual revenue**.

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

```bash
# Clone or navigate to the project directory
cd credit_card_analytics

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place the Kaggle dataset at:
```
data/raw/CC_GENERAL.csv
```

Download from: [Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

---

## How to Run

Execute the Jupyter notebooks in sequence:

```bash
cd notebooks

# Step 1: Data loading, cleaning, and EDA
jupyter notebook 01_EDA_and_Cleaning.ipynb

# Step 2: Feature engineering (6 derived ratios)
jupyter notebook 02_Feature_Engineering.ipynb

# Step 3: Normalisation, K-Means clustering, PCA visualisation
jupyter notebook 03_Clustering_and_PCA.ipynb

# Step 4: Anomaly detection (IQR + Isolation Forest)
jupyter notebook 04_Anomaly_Detection.ipynb

# Step 5: Business insights, persona report, Power BI exports
jupyter notebook 05_Business_Insights.ipynb
```

Each notebook loads the output of the previous one, so they **must be
run in order** (01 → 05).

---

## Project Structure

```
credit_card_analytics/
│
├── data/
│   ├── raw/                        # CC_GENERAL.csv (input)
│   ├── processed/                  # Cleaned, featured, master CSVs
│   └── exports/                    # Power BI-ready CSVs
│
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Clustering_and_PCA.ipynb
│   ├── 04_Anomaly_Detection.ipynb
│   └── 05_Business_Insights.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Data loading, cleaning, EDA plots
│   ├── features.py                 # 6 engineered feature ratios
│   ├── clustering.py               # StandardScaler, K-Means, PCA
│   ├── anomaly.py                  # IQR/Z-score + Isolation Forest
│   └── utils.py                    # Paths, constants, formatters
│
├── sql/
│   └── fraud_flag_rules.sql        # PostgreSQL fraud-tier logic
│
├── outputs/
│   ├── plots/                      # All saved figures (PNG + HTML)
│   └── reports/
│       └── segment_personas.md     # Business persona report
│
├── POWERBI_GUIDE.md                # Power BI dashboard setup guide
├── requirements.txt                # Pinned Python dependencies
└── README.md                       # This file
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Data manipulation | pandas 2.2, NumPy 1.26 |
| Machine learning | scikit-learn 1.5 (K-Means, PCA, Isolation Forest) |
| Visualisation | matplotlib 3.9, seaborn 0.13, Plotly 5.23 |
| Statistical analysis | SciPy 1.14 |
| Elbow detection | kneed 0.8 |
| Model persistence | joblib 1.4 |
| Database | PostgreSQL (SQL scripts provided) |
| Dashboarding | Power BI (CSV exports + DAX guide) |

---

## Resume-Ready Project Description

> Designed and implemented a customer segmentation and fraud detection
> analytics pipeline for a credit card portfolio of ~8,950 customers.
> Engineered 6 behavioural spending ratios and applied K-Means clustering
> (k=4) to identify four distinct customer personas — Transactors,
> Revolvers, Cash Advance Users, and Dormant accounts. Deployed Isolation
> Forest anomaly detection to flag 2.3% of customers as high-risk,
> generating a continuous Fraud Risk Score (0–100) per customer.
> Delivered actionable segment-specific strategies projected to yield
> ₹14+ lakh in incremental annual revenue through targeted upsell
> and retention campaigns. Built Power BI-ready data exports with
> DAX measures and PostgreSQL fraud-flagging views for production
> deployment.

---

## Outputs

| Output | Location |
|--------|----------|
| Cleaned dataset | `data/processed/cleaned_customers.csv` |
| Feature-enriched dataset | `data/processed/featured_customers.csv` |
| Master dataset (all features + scores) | `data/processed/master_customers.csv` |
| Power BI CSVs | `data/exports/` |
| EDA + clustering plots | `outputs/plots/` |
| Segment persona report | `outputs/reports/segment_personas.md` |
| SQL fraud rules | `sql/fraud_flag_rules.sql` |

---

## Currency & Locale

All monetary values, formatting, and business assumptions use
**Indian norms (₹)**. The `format_inr()` utility in `src/utils.py`
formats numbers with the Indian grouping system (e.g., ₹1,23,456.78).

---

## Licence

This project is for educational and portfolio purposes.
Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
under CC0 public domain licence.
