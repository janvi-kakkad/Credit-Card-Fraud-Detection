-- ============================================================
-- PROJECT  : Credit Card Spending Behaviour Analytics
-- AUTHOR   : [Your Name]
-- DATE     : 2026-04-16
-- DATABASE : PostgreSQL 15+
-- PURPOSE  : Fraud flagging rules, risk-tier classification,
--            and analytical queries for customer segmentation
--            data exported from the Python analytics pipeline.
--
-- RATIONALE:
--   Fraud tiers are derived from a combination of:
--     1. Isolation Forest anomaly scores (FRAUD_RISK_SCORE)
--     2. Cash-advance dominance (high CA ratio + utilisation)
--     3. Repayment delinquency (low payment-to-balance on
--        high outstanding balances)
--
--   The three-tier system (HIGH / MEDIUM / LOW) maps directly
--   to operational playbooks:
--     HIGH   → Immediate analyst review + account restriction
--     MEDIUM → Automated monitoring + monthly review
--     LOW    → Standard portfolio treatment
-- ============================================================


-- ────────────────────────────────────────────────────────────
-- 1. BASE TABLE
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customer_analytics (
    customer_id            VARCHAR(20) PRIMARY KEY,
    balance                NUMERIC(12, 2),
    balance_frequency      NUMERIC(5, 4),
    purchases              NUMERIC(12, 2),
    oneoff_purchases       NUMERIC(12, 2),
    installments_purchases NUMERIC(12, 2),
    cash_advance           NUMERIC(12, 2),
    purchases_frequency    NUMERIC(5, 4),
    oneoff_purchases_frequency     NUMERIC(5, 4),
    purchases_installments_frequency NUMERIC(5, 4),
    cash_advance_frequency NUMERIC(5, 4),
    cash_advance_trx       INTEGER,
    purchases_trx          INTEGER,
    credit_limit           NUMERIC(12, 2),
    payments               NUMERIC(12, 2),
    minimum_payments       NUMERIC(12, 2),
    prc_full_payment       NUMERIC(5, 4),
    tenure                 INTEGER,

    -- Engineered features
    utilization_ratio      NUMERIC(7, 4),
    payment_to_balance     NUMERIC(10, 4),
    spend_velocity         NUMERIC(7, 4),
    installment_frequency  NUMERIC(7, 4),
    cash_advance_ratio     NUMERIC(7, 4),
    purchase_type_ratio    NUMERIC(7, 4),

    -- Clustering
    cluster_id             SMALLINT,
    segment_name           VARCHAR(30),

    -- Anomaly detection
    anomaly_score          NUMERIC(10, 6),
    anomaly_flag           SMALLINT,
    fraud_risk_score       NUMERIC(6, 2),

    -- Classic outlier flags
    balance_outlier        BOOLEAN,
    purchase_outlier       BOOLEAN,
    cash_outlier           BOOLEAN,
    classic_outlier_flag   SMALLINT,

    -- PCA coordinates
    pc1                    NUMERIC(10, 6),
    pc2                    NUMERIC(10, 6)
);


-- ────────────────────────────────────────────────────────────
-- 2. FRAUD RISK VIEW
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW fraud_risk_view AS
SELECT
    customer_id,
    segment_name,
    fraud_risk_score,

    -- Three-tier fraud classification
    CASE
        WHEN cash_advance_ratio > 0.6
             AND utilization_ratio > 0.85
            THEN 'HIGH_RISK'
        WHEN fraud_risk_score > 75
            THEN 'HIGH_RISK'
        WHEN fraud_risk_score BETWEEN 50 AND 75
            THEN 'MEDIUM_RISK'
        WHEN payment_to_balance < 0.05
             AND balance > 3000
            THEN 'MEDIUM_RISK'
        ELSE 'LOW_RISK'
    END AS fraud_tier,

    -- Over-limit flag: zero full payments AND balance near limit
    CASE
        WHEN prc_full_payment = 0
             AND balance > credit_limit * 0.9
            THEN 1
        ELSE 0
    END AS overlimit_flag,

    -- Cash-dominant flag: cash advances exceed 2× purchases
    CASE
        WHEN cash_advance > purchases * 2
            THEN 1
        ELSE 0
    END AS cash_dominant_flag

FROM customer_analytics;


-- ────────────────────────────────────────────────────────────
-- Q1. Count and % of customers per fraud tier per segment
-- ────────────────────────────────────────────────────────────
-- Purpose: Identifies which segments concentrate the most
--          high-risk customers for targeted intervention.
SELECT
    segment_name,
    fraud_tier,
    COUNT(*)                                      AS customer_count,
    ROUND(
        COUNT(*) * 100.0
        / SUM(COUNT(*)) OVER (PARTITION BY segment_name),
        2
    )                                             AS pct_within_segment
FROM fraud_risk_view
GROUP BY segment_name, fraud_tier
ORDER BY segment_name, fraud_tier;


-- ────────────────────────────────────────────────────────────
-- Q2. Average fraud risk score and utilisation by segment
-- ────────────────────────────────────────────────────────────
-- Purpose: Quick executive summary of risk exposure per
--          customer cohort, sorted by decreasing risk.
SELECT
    ca.segment_name,
    COUNT(*)                                      AS customer_count,
    ROUND(AVG(ca.fraud_risk_score), 2)            AS avg_fraud_score,
    ROUND(AVG(ca.utilization_ratio), 4)           AS avg_utilization,
    ROUND(AVG(ca.cash_advance_ratio), 4)          AS avg_ca_ratio
FROM customer_analytics ca
GROUP BY ca.segment_name
ORDER BY avg_fraud_score DESC;


-- ────────────────────────────────────────────────────────────
-- Q3. Revolvers who are also HIGH_RISK (priority list)
-- ────────────────────────────────────────────────────────────
-- Purpose: These customers carry high balances, pay little,
--          and exhibit anomalous behaviour — ideal candidates
--          for proactive outreach or account restriction.
SELECT
    frv.customer_id,
    frv.fraud_risk_score,
    ca.balance,
    ca.utilization_ratio,
    ca.cash_advance_ratio,
    ca.payment_to_balance
FROM fraud_risk_view frv
JOIN customer_analytics ca
    ON frv.customer_id = ca.customer_id
WHERE frv.segment_name = 'Revolvers'
  AND frv.fraud_tier   = 'HIGH_RISK'
ORDER BY frv.fraud_risk_score DESC;


-- ────────────────────────────────────────────────────────────
-- Q4. Monthly payment behaviour comparison (window functions)
-- ────────────────────────────────────────────────────────────
-- Purpose: Ranks customers within each segment by payment
--          discipline (payment_to_balance ratio), enabling
--          percentile-based tiering for retention campaigns.
SELECT
    ca.customer_id,
    ca.segment_name,
    ca.payments,
    ca.balance,
    ca.payment_to_balance,
    ROUND(
        PERCENT_RANK() OVER (
            PARTITION BY ca.segment_name
            ORDER BY ca.payment_to_balance
        ), 4
    )                                             AS pmt_percentile_rank,
    NTILE(10) OVER (
        PARTITION BY ca.segment_name
        ORDER BY ca.payment_to_balance
    )                                             AS pmt_decile,
    ROUND(
        AVG(ca.payment_to_balance) OVER (
            PARTITION BY ca.segment_name
        ), 4
    )                                             AS segment_avg_pmt_ratio
FROM customer_analytics ca
ORDER BY ca.segment_name, pmt_percentile_rank DESC;
