# AI SaaS Growth Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/ML-Gradient%20Boosting-green)
![AI](https://img.shields.io/badge/AI-Gemini%202.5-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> An end-to-end customer retention intelligence platform
> that identifies at-risk customers, quantifies revenue
> exposure, and generates AI-powered recommendations —
> built from raw data to deployed web application.

---


## Live Demo

🚀 **[Launch the App →](https://saas-growth-intelligence-udj9ecuqzzw6oh8qcbbbmw.streamlit.app)**

Upload your own customer CSV or explore the built-in
demo analysis of 7,032 real SaaS customers.

---

## What Makes This Different

Most churn analysis projects on GitHub produce
an accuracy score and stop there.

This system produces a complete retention
intelligence platform:

| What others build | What this builds |
|---|---|
| Churn prediction (yes/no) | Risk score 0–100 per customer |
| Overall churn rate | Revenue at risk per segment |
| Basic EDA charts | Survival curves + cohort heatmaps |
| Accuracy metrics | Business dollar impact |
| Static notebook | Deployed interactive web app |
| Telco dataset only | Dashboard analytics for compatible customer CSVs |
| Manual reporting | Auto-generated executive briefing |

Built across 14 phases with full business
reasoning documented at every decision point.


## The Business Problem

A SaaS telecommunications company was losing
**26.58% of its customers annually** with no systematic
way to identify who would leave, why they were leaving,
or what the business should do about it.

This represents **$1,669,570 in preventable annual
revenue loss** — $139,131 every month leaving silently.

The business had data. It had no intelligence.

This system converts raw customer data into a
complete retention intelligence platform.

---

## What This System Does

| Capability | Description |
|---|---|
| Churn Prediction | Gradient Boosting model — 76.74% recall |
| Risk Scoring | Every customer scored 0–100 with risk level |
| Driver Analysis | Top 3 churn reasons identified per customer |
| Revenue Intelligence | Dollar impact calculated per segment |
| Survival Analysis | Kaplan-Meier retention curves by contract type |
| Cohort Analysis | Retention heatmap across tenure cohorts |
| SaaS KPI Engine | MRR, ARR, LTV, CAC, LTV:CAC ratio |
| Executive Briefing | Auto-generated summary from analysis outputs |
| Interactive App | 5-page Streamlit dashboard — no code needed |
| Data Upload | Validated dashboard analytics for compatible customer CSVs |

---

## Demo Analysis — Telco Dataset Results

> The following results are from the built-in demo
> analysis of 7,032 telecom SaaS customers.
> Upload your own data to see results for your business.

### Business Overview

| Metric | Value | Benchmark | Status |
|---|---|---|---|
| Annual Churn Rate | 26.58% | < 7% | 🔴 Critical |
| Monthly Revenue (MRR) | $316,530 | — | — |
| Annual Revenue (ARR) | $3,798,362 | — | — |
| Monthly Revenue Lost | $139,131 | — | 🔴 At Risk |
| Annual Revenue Lost | $1,669,570 | — | 🔴 At Risk |
| Customer LTV | $1,166 | — | — |
| LTV:CAC Ratio | 3.89:1 | > 3.0:1 | 🟡 Marginal |
| Revenue Retention | 69.47% | > 85% | 🔴 Critical |

### Key Finding 1 — Contract Type

| Contract | Churn Rate | vs Overall |
|---|---|---|
| Month-to-month | 42.71% | 1.6× above |
| One year | 11.28% | Below average |
| Two year | 2.85% | 9.3× below |

Month-to-month customers churn at **15× the rate**
of two-year contract customers — the single
strongest predictor in the entire dataset.

### Key Finding 2 — Customer Lifecycle

| Tenure Group | Churn Rate | Customers |
|---|---|---|
| New (0–12 months) | 47.68% | 2,175 |
| Developing (13–24 months) | 28.71% | 1,024 |
| Established (25–48 months) | 20.39% | 1,594 |
| Loyal (49–72 months) | 9.51% | 2,239 |

New customers churn at **5× the rate** of loyal
customers. The 0–6 month cohort alone represents
**38.1% of total annual revenue risk**.


Cohort deep-dive: within the New (0–12 month) group,
the 0–6 month cohort alone generates **38.1% of total
annual revenue risk** despite representing only 20.9%
of all customers — making early-stage onboarding the
single highest-leverage retention intervention.


### Key Finding 3 — Payment Method

| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.29% |
| Mailed check | 19.20% |
| Bank transfer (automatic) | 16.73% |
| Credit card (automatic) | 15.25% |

Automatic payment customers churn at **3× lower rates**.
Payment method is a proxy for customer commitment level.

### Key Finding 4 — High Risk Profile

The following customer combination produces
maximum churn probability (74%+ churn rate):
Month-to-month contract

Tenure under 12 months
Electronic check payment
Monthly charges $80–100

### Key Finding 5 — Survival Analysis

Probability of customer still being active:
          Month 12    Month 36    Month 60
Month-to-month  70.3%      49.1%      29.7%
One year        99.1%      95.9%      83.2%
Two year       100.0%      99.9%      98.6%

Gap at month 12: 29.7 percentage points between
month-to-month and two-year contracts.

### ML Model Performance

Three models trained and compared on 1,407 test customers:

| Model | Recall | AUC-ROC | Churners Caught |
|---|---|---|---|
| Logistic Regression | 69.52% | 83.23% | 260 / 374 |
| Random Forest | 72.46% | 83.23% | 271 / 374 |
| **Gradient Boosting** | **76.74%** | **83.18%** | **287 / 374** |
| Baseline (no model) | — | — | 0 / 374 |

**Recall improved from 52.67% (default Logistic Regression)
to 76.74% (tuned Gradient Boosting) — a +24 percentage
point improvement through three techniques:
class_weight='balanced' for class imbalance,
threshold tuning from 0.50 to 0.29, and
Gradient Boosting over simpler classifiers.

**Why recall is the priority metric:**
Missing a churner costs ~$1,166/year (customer LTV).
Contacting a false positive costs ~$5–10.
The 80:1 cost asymmetry makes maximising recall
the correct business objective.

### Risk Scoring Output

Every customer receives a complete profile:

Every customer receives a complete profile:
Customer ID      : 9300-AGZNL
Risk Score       : 96.2 / 100
Risk Level       : CRITICAL
Churn Probability: 96.2%
Revenue at Risk  : $94.00/month | $1,128.00/year
Priority Rank    : #1 of 7,032
Why at risk:
→ Month-to-month contract
→ Manual payment method
→ New customer (0-12 months)
Recommended Action:
• Personal call within 24 hours
• Offer 40% discount on annual plan
• Escalate to senior retention agent
Actual Outcome   : Churned ✓ (model was correct)

**Critical risk customers show 84.0% actual churn
rate — the system is 3.2× more accurate than random.**

### Recommended Interventions

**Intervention 1 — Contract migration campaign
Target 3,875 month-to-month customers with
time-limited annual plan upgrade incentives.
Projected savings: $145,411/year at 10% conversion.

**Intervention 2 — 90-day onboarding programme**
Structured touchpoints at weeks 2, 4, 8, and 12
for all new customers. Target activation of 3+
services in first 30 days. Addresses the 0–6 month
cohort that generates 38.1% of total revenue risk.

**Intervention 3 — Automatic payment incentive**
Offer 10% monthly discount for switching to
automatic payment at signup. Reduces churn signal
from 45.29% toward 15.25%.

**Combined 30% churn reduction would save
$436,234 annually.**

---

## Executive Briefing

The system automatically generates a written briefing
from all analysis outputs using Gemini 2.5 Flash.

> *"Our company is currently experiencing an annual
> churn rate of 26.58%, resulting in a significant
> annual revenue loss of $1,669,570. While our LTV
> to CAC ratio stands at a healthy 3.89:1, exceeding
> the 3.0 minimum benchmark, the substantial churn
> rate indicates a critical area for improvement...
> A critical segment of 973 customers currently faces
> an alarming 84.0% churn rate. This high-risk group
> alone places $939,869 in revenue at immediate risk,
> demanding urgent attention..."*

A CEO reads this in 60 seconds and knows exactly
what to do. No notebook required.

---

## Streamlit App — 5 Pages
Page 0 — Data Upload
Upload your own CSV or load demo data
Instant validation + quick analysis preview
Page 1 — Executive Dashboard
Business health score · AI briefing
Risk distribution · Top 10 priority customers
Page 2 — Customer Risk Explorer
Search any customer by ID
Full risk profile · Gauge chart
What-if simulator (contract/payment changes)
Page 3 — Segment Analysis
Live filters: contract · payment · tenure
4 interactive charts update in real time
Engagement vs churn analysis
Page 4 — Revenue Calculator
Slider: set churn reduction target
Revenue saved · New MRR · ROI calculator
Payback period · All scenarios table
Page 5 — Recommendation Engine
AI segment strategy cards
Customer action list with filters
Download CSV for retention team

---

## Project Structure
saas-growth-intelligence/
│
├── data/
│   ├── raw/                        ← source dataset
│   └── cleaned/
│       ├── telco_churn_clean.csv
│       ├── features_matrix.csv
│       └── customer_risk_profiles.csv
│
├── notebooks/                      ← 13 analysis notebooks
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_sql_analysis.ipynb
│   ├── 05_eda.ipynb
│   ├── 06_saas_kpi_analysis.ipynb
│   ├── 07_survival_analysis.ipynb
│   ├── 07b_cohort_analysis.ipynb
│   ├── 08_customer_segmentation.ipynb
│   ├── 09_feature_engineering.ipynb
│   ├── 10_ml_churn_model.ipynb
│   ├── 11_risk_scoring.ipynb
│   └── 12_ai_briefing_generator.ipynb
│
├── reports/
│   ├── figures/                    ← 12 portfolio charts
│   ├── executive_briefing.txt
│   └── segment_recommendations.json
│
├── app/
│   └── streamlit_app.py            ← 5-page web app
│
├── models/                         ← trained ML models
├── src/                            ← utility functions
├── requirements.txt
├── .gitignore
└── README.md

---

## Technical Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Analysis | Pandas, NumPy, SQLite3 |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (LR, RF, Gradient Boosting) |
| Survival Analysis | Custom Kaplan-Meier implementation |
| AI Integration | Google Gemini 2.5 Flash API |
| Web Application | Streamlit |
| Version Control | Git + GitHub |
| Environment | Virtual environment + python-dotenv |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/niviya-albert/saas-growth-intelligence.git
cd saas-growth-intelligence

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
# Create .env file in project root:
# GEMINI_API_KEY=your_key_here

# 5. Get the dataset
# Download from Kaggle (free):
# kaggle.com/datasets/blastchar/telco-customer-churn
# Save as: data/raw/telco_churn_raw.csv

# 6. Run notebooks in order
# Open notebooks/ folder in VS Code
# Run 01 through 12 sequentially
# This generates all data files and models

# 7. Launch the app
streamlit run app/streamlit_app.py
```

---

## Use With Your Own Data

The system accepts any customer CSV with these
minimum columns:

| Column | Description |
|---|---|
| customerID | Unique customer identifier |
| tenure | Months as a customer |
| MonthlyCharges | Monthly payment amount |
| TotalCharges | Cumulative payments |
| Contract | Contract type |
| PaymentMethod | How they pay |
| Churn | Did they leave? (Yes/No or 1/0) |

Upload via the Data Upload page in the app.
For a fully customised implementation on your
business data, contact me directly.

---

## About This Project

Built as a portfolio project demonstrating
complete data analytics capability — from
business problem framing through deployed
AI-powered web application.

**14 phases completed:**
Business understanding → Data cleaning →
SQL analysis → EDA → SaaS KPI analysis →
Survival analysis → Cohort analysis →
Customer segmentation → Feature engineering →
Machine learning → Risk scoring →
AI briefing generation → Streamlit deployment →
GitHub documentation

**Skills demonstrated:**
SQL · Python · Statistical analysis ·
Machine learning · Survival analysis ·
Cohort analysis · Feature engineering ·
Risk scoring · AI API integration ·
Streamlit development · Business intelligence ·
Executive communication · Professional documentation

---

## License

This project is for portfolio and educational purposes.
Commercial use requires written permission.

© 2026 Niviya Albert · niviyalbert1365@gmail.com

---

*Dataset: IBM Telco Customer Churn via Kaggle*
*Powered by Python · Scikit-learn · Gemini AI · Streamlit*
