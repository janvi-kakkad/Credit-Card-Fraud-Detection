# Credit Card Customer Segmentation — Persona Report

## Executive Summary

Analysis of 8,949 credit card customers using K-Means clustering (k=4) identified four distinct behavioural segments. Isolation Forest anomaly detection (contamination=2.3%) flagged 206 customers (2.3%) as potentially anomalous. The **Revolvers** segment, while carrying the highest default risk, is the primary interest-revenue driver. Targeted upsell and retention strategies across segments can yield an estimated incremental revenue of ₹4,09,172.98 annually.

## Methodology

- **Clustering:** K-Means with k=4 on 10 normalised features (5 original + 5 engineered behavioural ratios).
- **Normalisation:** StandardScaler (zero-mean, unit-variance).
- **Anomaly Detection:** Isolation Forest with contamination=0.023 (200 estimators, random_state=42).
- **Dimensionality Reduction:** PCA (2 components) for visualisation.
- **Currency:** All monetary values in Indian Rupees (₹).

## Segment Profiles

### 1. Transactors (Cluster 3 — ~15% of customers)

**Behavioural Profile:** High-frequency purchasers who pay off their balances regularly. They use their credit cards primarily for transactions rather than as a borrowing instrument.

**Key Metrics:** Avg balance: ₹1,567.76 | Avg purchases: ₹3,695.87 | Utilisation: 41.2% | Full payment rate: 22.8%

**Business Insight:** These customers generate low interest revenue but contribute significantly through transaction fees and merchant interchange. They represent the healthiest portfolio segment.

**Recommended Action:** Upsell premium rewards cards (e.g., travel miles, tiered cashback). Offer credit limit increases to encourage higher spend volume. Target with co-branded merchant partnerships for increased swipe frequency.

**Fraud Risk:** Generally low — disciplined payment behaviour keeps anomaly scores minimal.

### 2. Revolvers (Cluster 2 — ~36% of customers)

**Behavioural Profile:** Carry high outstanding balances month-over-month with minimal repayment. High credit utilisation ratios indicate they operate near their credit limits consistently.

**Key Metrics:** Avg balance: ₹1,661.39 | Avg purchases: ₹200.89 | Utilisation: 62.7% | Full payment rate: 3.3%

**Business Insight:** Highest value from an interest revenue perspective — these customers are the primary profit drivers for the card portfolio. However, they also carry elevated default risk. Retention is a priority.

**Recommended Action:** Offer balance consolidation products and EMI conversion at competitive rates. Flag for early intervention if utilisation exceeds 90%. Consider proactive restructuring offers before delinquency.

**Fraud Risk:** Medium — elevated balance with low repayment warrants continuous monitoring.

### 3. Cash Advance Users (Cluster 0 — ~10% of customers)

**Behavioural Profile:** Rely heavily on cash advances (ATM withdrawals against credit limit) rather than point-of-sale purchases. This behaviour is often associated with financial distress or liquidity crunches.

**Key Metrics:** Avg balance: ₹5,875.95 | Avg purchases: ₹951.48 | Utilisation: 61.9% | Full payment rate: 3.5%

**Business Insight:** Highest fraud risk cohort. Cash advance dominance is a documented precursor to both fraud and default. These customers may also be under financial stress, requiring careful handling.

**Recommended Action:** Restrict cash advance limits proactively. Trigger manual review when cash_advance_ratio exceeds 0.6. Consider financial counselling outreach and alternative lending products.

**Fraud Risk:** High — cash advance dominance is a documented fraud precursor.

### 4. Dormant/Low Engagement (Cluster 1 — ~39% of customers)

**Behavioural Profile:** Minimal activity across all spending and payment dimensions. These cardholders rarely use their credit cards and maintain low balances.

**Key Metrics:** Avg balance: ₹404.99 | Avg purchases: ₹700.47 | Utilisation: 9.6% | Full payment rate: 26.5%

**Business Insight:** Low revenue contribution with significant churn risk. The cost of maintaining inactive accounts may exceed their marginal revenue, making activation campaigns essential.

**Recommended Action:** Launch activation campaigns with bonus reward points for first purchase. Deploy personalised spend-category offers based on TENURE. Consider account closure for long-dormant accounts to reduce portfolio cost.

**Fraud Risk:** Low overall — but inactivity followed by sudden high-value spend should be flagged for review (potential account takeover pattern).

## Revenue Impact Estimates

### Transactor Upsell Revenue

- Eligible Transactors: 1,366
- Assumed conversion rate: 15%
- Avg annual spend increase per converted customer: ₹25,000
- Net margin on incremental spend: 3.5%
- **Formula:** 1,366 × 0.15 × ₹25,000 × 0.035
- **Estimated revenue:** ₹1,79,287.50

### Revolver Retention Revenue

- Revolver count: 3,203
- Avg outstanding balance: ₹1,661.39
- Monthly interest rate: 3.6%
- Retention improvement: 10% fewer churns
- **Formula:** 3,203 × 0.10 × Avg Balance × 0.036 × 12
- **Estimated revenue:** ₹2,29,885.48

### Total Estimated Incremental Revenue: ₹4,09,172.98

## Anomaly Detection Summary

- **Total customers flagged:** 206 (2.3% of dataset)

- **Breakdown by segment:**

  - Transactors: 87 anomalies (6.4% of segment)

  - Revolvers: 4 anomalies (0.1% of segment)

  - Cash Advance Users: 109 anomalies (12.5% of segment)

  - Dormant/Low Engagement: 6 anomalies (0.2% of segment)

- **Top risk indicators:** High cash advance ratio (>0.6), high utilisation (>0.85), near-zero payment-to-balance ratio, and sudden high-value transactions on dormant accounts.
