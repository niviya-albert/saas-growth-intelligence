# AI SaaS Growth Intelligence System

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/ML-Gradient%20Boosting%20%7C%20Ensemble-green)
![AI](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-orange)
![Tests](https://img.shields.io/badge/Tests-Pytest%20100%25%20Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-86%25-success)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blueviolet)

> A customer churn analysis that actually answers the question a retention manager would ask: *which specific customers do I contact tomorrow, and what do I say to them?*

---

## Live Demo

🚀 **[Launch the Interactive App →](https://saas-growth-intelligence-udj9ecuqzzw6oh8qcbbbmw.streamlit.app)**

Upload your own customer data or explore the built-in demo of 7,032 telecom accounts.

---

## Why I Built This

Most churn prediction projects I came across had the same structure: load the Telco dataset, train a model, get ~80% accuracy, done.

But accuracy doesn't tell a retention team anything useful. It doesn't say *who* to call, *why* that specific customer is at risk, or *what the financial impact* of losing them would be.

I built this to close that gap — to make the output of the model something a non-technical manager could actually act on the same day they opened the dashboard.

---

## What Makes This Different

Most churn analysis projects on GitHub produce an accuracy score and stop there. This system delivers a complete, production-grade retention intelligence platform:

| Standard Churn Projects | This System |
|---|---|
| Binary classification (yes/no) | Risk score 0–100 per customer with weighted ensemble |
| Overall churn percentage | Exact annual ARR revenue at risk per segment |
| Basic exploratory charts | Kaplan-Meier survival curves & tenure cohort heatmaps |
| Static confusion matrix | Business dollar impact, ROI, & payback period modeler |
| Unstructured notebook script | Modular production package (`src/`), Pytest suite, CI/CD |
| Static bundled dataset only | Real-time batch scoring & dynamic What-If simulator on custom CSV uploads |
| Manual summary text | Live Gemini 2.5 Flash executive briefing generator |

---

## The Business Problem

A SaaS telecommunications provider was losing **26.58% of its customers annually** with no systematic method to identify who would leave, why they were leaving, or what interventions would maximize retention.

- **Total Preventable Loss:** **$1,669,570 in annual revenue** ($139,131 monthly revenue bleeding silently across 1,869 churned accounts).
- **Target Financial Recovery:** A 10%–30% targeted intervention success rate protects **$145,321 to $435,964 in recurring ARR annually**.

This platform transforms raw customer event and billing data into an automated, actionable retention engine.

---

## System Capabilities

| Capability | Description |
|---|---|
| **Ensemble Prediction** | Tuned Gradient Boosting + Random Forest + Logistic Regression (76.74% recall) |
| **Risk Scoring Engine** | Every customer scored 0–100 with 4 risk tiers (`Critical`, `High`, `Medium`, `Low`) |
| **Driver Attribution** | Automated extraction of top 3 churn reasons per account |
| **What-If Simulator** | Live model re-scoring on contract, billing, or tenure adjustments |
| **Revenue Intelligence** | Quantifies monthly and annual revenue exposure per segment |
| **Survival Analysis** | Kaplan-Meier retention trajectories by contract commitment |
| **SaaS KPI Engine** | Real-time MRR, ARR, LTV, CAC, LTV:CAC, and Net Revenue Retention |
| **AI Briefing Generator** | Automated executive briefings powered by Google Gemini 2.5 Flash |
| **Interactive Dashboard** | 6-page responsive Streamlit application |
| **Custom CSV Ingestion** | End-to-end schema validation, feature engineering, and live batch scoring |

---

## Demo Analysis — Key Findings (7,032 Telco Accounts)

### Business Health & Unit Economics

| Metric | Value | Benchmark | Status |
|---|---|---|---|
| Annual Churn Rate | 26.58% | < 7% | 🔴 Critical |
| Monthly Active Revenue (MRR) | $316,530 | — | — |
| Annual Active Revenue (ARR) | $3,798,362 | — | — |
| Annual Revenue Lost | $1,669,570 | — | 🔴 At Risk |
| Customer Lifetime Value (LTV) | $1,166 | — | — |
| LTV : CAC Ratio | 3.89 : 1 | > 3.0 : 1 | 🟡 Marginal |
| Net Revenue Retention | 69.47% | > 85% | 🔴 Critical |

### Portfolio Risk Tier Breakdown (7,032 Accounts)

| Risk Tier | Score Range | Accounts | % Base | Exposed ARR | Recommended SLA |
|---|---|---|---|---|---|
| **Critical** | 75–100 | **937** | 13.3% | **$908,609** | 24-hr personal CSM intervention |
| **High** | 50–74 | 864 | 12.3% | $542,180 | 3-day automated discount / billing fix |
| **Medium** | 25–49 | 1,421 | 20.2% | $784,520 | 90-day onboarding nurture sequence |
| **Low** | 0–24 | 3,810 | 54.2% | $1,563,053 | Loyalty perks & annual renewal sync |

> **Concentrated Exposure:** **54.4% of all revenue risk ($908.6K)** is concentrated in just **13.3% of customers (937 accounts)** in the Critical tier.

### Primary Churn Drivers & Feature Stickiness

1. **Contract Type Commitment**: Month-to-month accounts churn at **42.71%** vs **2.85%** for two-year contracts (a 15× risk multiple).
2. **Onboarding Lifecycle Gap**: Accounts in months 0–6 represent **38.1% of total annual revenue risk** despite being only 20.9% of the customer base.
3. **Payment Friction**: Manual payment methods (Electronic checks) churn at **45.29%** compared to **15.25%** for automated credit card billing.
4. **Product Stickiness (Add-ons)**: Accounts without **Tech Support & Online Security** churn at **41.6%**, compared to **14.6%** for accounts with security services attached (a 2.8× retention lift).

### Retention Financial Impact & ROI Modeler

Modeled on a $50,000 targeted retention intervention program:

| Churn Reduction Target | Churned Accounts Saved | Annual ARR Protected | Net Annual Gain ($50K Cost) | ROI % | Payback Period |
|---|---|---|---|---|---|
| **10% Reduction** | 187 accounts | **$145,321** | $95,321 | 190.6% | 4.1 months |
| **20% Reduction** | 374 accounts | **$290,643** | **$240,643** | **481.3%** | **2.1 months** |
| **30% Reduction** | 561 accounts | **$435,964** | **$385,964** | **771.9%** | **1.4 months** |

### ML Model Performance & Cost Asymmetry

| Model | Recall | AUC-ROC | Churners Caught (Test Set) |
|---|---|---|---|
| Logistic Regression (Tuned) | 69.52% | 83.23% | 260 / 374 |
| Random Forest (Regularized) | 75.94% | 83.75% | 284 / 374 |
| **Gradient Boosting (Tuned)** | **76.74%** | **84.07%** | **287 / 374** |

> **Why Recall is Priority:** Missing a churner costs ~$1,166 (customer LTV). Contacting a false positive costs ~$5–10. An 80:1 cost asymmetry makes maximizing recall the optimal financial strategy.

---

## Streamlit Web App — 6 Interactive Pages

- **Page 0 — Data Upload & Live ML Risk Scoring**: Upload custom CSVs for instant schema validation, 30-feature extraction, and dynamic ensemble risk scoring.
- **Page 1 — Executive Dashboard**: Business Health Score gauge, active ARR at risk, portfolio risk breakdown, and optional live Gemini AI executive briefing generation.
- **Page 2 — Customer Risk Explorer & Dynamic Simulator**: Account search with real-time What-If parameter modification (contract/payment) querying the live ML ensemble model.
- **Page 3 — Segment Analysis**: Multi-dimensional filters (contract, payment, tenure) updating dynamic Plotly visualisations in real time.
- **Page 4 — Revenue Impact Calculator & ROI Modeler**: Churn reduction target slider, ARR savings projections, and retention program payback period calculator.
- **Page 5 — Prescriptive Retention Playbook**: Strategic playbooks by risk tier and exportable outreach queue (CSV).

---

## Project Structure

```
saas-growth-intelligence/
│
├── .github/workflows/
│   └── ci.yml                      ← Automated CI/CD pipeline (lint, test, build)
│
├── src/                            ← Production Modular Python Package
│   ├── data/
│   │   ├── cleaner.py              ← Schema validation & cleaning pipeline
│   │   └── loader.py               ← Dataset resolution & path loaders
│   ├── features/
│   │   └── engineering.py          ← Leakage-free 30-feature transformer
│   ├── metrics/
│   │   └── saas_kpis.py            ← MRR, ARR, LTV, CAC, Health Score engine
│   ├── models/
│   │   ├── train.py                ← LR, RF, Gradient Boosting & threshold tuner
│   │   └── predict.py              ← Inference, ensemble scoring & What-If simulator
│   ├── llm/
│   │   └── briefing.py             ← Google Gemini 2.5 Flash briefing engine
│   └── pipeline.py                 ← End-to-end CLI orchestrator
│
├── tests/                          ← Comprehensive Pytest Suite (86%+ coverage)
│   ├── test_cleaner.py             ← Data validation & sanitization tests
│   ├── test_features.py            ← Feature matrix transformation tests
│   ├── test_kpis.py                ← Unit economic & health score math tests
│   ├── test_models.py              ← Inference & What-If simulation tests
│   └── test_pipeline.py            ← End-to-end integration tests
│
├── app/
│   └── streamlit_app.py            ← 6-page interactive web application
│
├── data/
│   ├── raw/                        ← Source dataset (telco_churn_raw.csv)
│   └── cleaned/                    ← Cleaned data & features matrix
│
├── models/                         ← Compressed trained models (<800KB total)
├── notebooks/                      ← 13 exploratory & analytical notebooks (01-12)
├── reports/                        ← Executive briefings, recommendations & figures
├── requirements.txt                ← Production dependency manifest (UTF-8)
├── SETUP.md                        ← Quickstart & environment guide
└── README.md
```

---

## Technical Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **Data Engineering** | Pandas, NumPy, SQLite3 |
| **Visualisation** | Matplotlib, Seaborn, Plotly Express & Graph Objects |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| **Survival & Cohorts** | Kaplan-Meier survival curves & retention lifecycle analysis |
| **AI / LLM Integration** | Google Gemini 2.5 Flash API (`google-genai` SDK) |
| **Web Application** | Streamlit |
| **Testing & CI/CD** | Pytest, Pytest-Cov, Ruff, GitHub Actions |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/niviya-albert/saas-growth-intelligence.git
cd saas-growth-intelligence

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows (PowerShell)
source .venv/bin/activate      # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite (19 unit & integration tests)
pytest --cov=src -v

# 5. Run end-to-end pipeline (reproduces data, models, and reports in ~4s)
python -m src.pipeline --skip-llm

# 6. Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

---

## Use With Your Own Data

The system accepts any customer subscription CSV with these columns:

| Column | Description | Type |
|---|---|---|
| `customerID` | Unique account identifier | String |
| `tenure` | Months as a customer | Integer |
| `MonthlyCharges` | Current monthly bill amount | Float |
| `TotalCharges` | Cumulative lifetime payments | Float |
| `Contract` | Commitment tier (`Month-to-month`, `One year`, `Two year`) | String |
| `PaymentMethod` | Billing channel (`Electronic check`, `Credit card`, etc.) | String |
| `Churn` | Observed attrition status (`Yes`/`No` or `1`/`0`) | String/Int |

Upload via **Data Upload & Scoring** in the web app for instant automated batch risk scoring and interactive What-If scenario modeling.

---

## Something That Surprised Me

I expected contract type to be the strongest churn predictor — and it was — but the interaction I didn't anticipate was within the fiber optic segment.

Month-to-month customers on **fiber optic** churn at **41.9%**, nearly double the 22.1% rate for DSL month-to-month customers on the same contract. That pattern doesn't show up if you just look at contract type alone.

It suggests either a pricing problem or a product quality issue specific to the fiber tier — something a purely ML-focused analysis would have completely missed without digging into the segment breakdown.

---

## Limitations

Things this project genuinely can't do well:

- **The dataset is from 2013.** Payment methods, contract norms, and customer expectations have changed. The model would need retraining on current data before being used in production.
- **LTV/CAC calculations use an estimated CAC.** Real implementations need your actual acquisition cost data, which varies by channel.
- **The What-If simulator holds everything else constant.** In reality, switching a customer from electronic check to auto-pay likely correlates with other behavioural changes the model doesn't capture.
- **The AI briefing is only as good as the input metrics.** If your churn labelling is inconsistent, the briefing will sound confident about wrong numbers.
- **It was built on one industry (telecom).** Column names, contract structures, and what counts as "churn" vary significantly across SaaS verticals.

---

## License

MIT License — free to use for learning, portfolio, or adaptation to your own data.

© 2026 Niviya Albert · [LinkedIn](https://www.linkedin.com/in/niviya-albert-271a5333a/) · niviyalbert1365@gmail.com
