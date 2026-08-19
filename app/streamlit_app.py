"""SaaS Growth Intelligence System - Streamlit Dashboard Application.

Interactive customer retention intelligence platform powered by Scikit-Learn
ensemble models and Google Gemini AI.
"""

import pathlib
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is in python path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.cleaner import clean_customer_data
from src.data.loader import load_clean_data, load_risk_profiles
from src.llm.briefing import (
    build_metrics_summary,
    generate_executive_briefing,
    load_saved_briefing_reports,
)
from src.metrics.saas_kpis import (
    calculate_reduction_scenarios,
    calculate_retention_roi,
    compute_business_health_score,
    compute_saas_kpis,
)
from src.models.predict import CustomerRiskScorer

# ── PAGE CONFIGURATION ───────────────────────────────────────
st.set_page_config(
    page_title="SaaS Growth Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── FONTS ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── SIDEBAR ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #818cf8 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(99, 102, 241, 0.25) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

/* ── PAGE BACKGROUND ─────────────────────────────────── */
.stApp {
    background: #f8fafc;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ── HEADINGS ────────────────────────────────────────── */
h1 {
    font-weight: 700;
    font-size: 1.9rem !important;
    color: #0f172a !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-weight: 600;
    color: #1e293b !important;
    letter-spacing: -0.3px;
}
h3, h4 {
    font-weight: 600;
    color: #334155 !important;
}

/* ── METRIC CARDS ────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
    transition: box-shadow 0.2s, transform 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.12);
    transform: translateY(-2px);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── CONTENT CARDS ───────────────────────────────────── */
.report-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.report-card:hover {
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.1);
    border-color: rgba(99, 102, 241, 0.3);
}
.report-card h4 {
    color: #1e293b !important;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 10px;
}
.report-card p, .report-card li {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.6;
}
.report-card ul {
    padding-left: 1.2rem;
    margin-top: 8px;
}
.report-card li {
    margin-bottom: 4px;
}

/* ── ACCENT DIVIDER ──────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}

/* ── BUTTONS ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 20px;
    letter-spacing: 0.01em;
    transition: box-shadow 0.2s, transform 0.15s;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.35);
}
.stButton > button:hover {
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.5);
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── DATAFRAMES ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}

/* ── TABS ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 500;
    color: #64748b;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.08);
}

/* ── ALERTS & INFO BOXES ─────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px;
    border-left-width: 4px;
}

/* ── SELECTBOX & INPUTS ──────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    border-radius: 10px !important;
    border-color: #e2e8f0 !important;
    font-size: 0.9rem;
}

/* ── FILE UPLOADER ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 14px;
}
[data-testid="stFileUploader"] > div {
    border: 2px dashed #c7d2fe !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #eef2ff 0%, #f8faff 100%) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #6366f1 !important;
    background: linear-gradient(135deg, #e0e7ff 0%, #eef2ff 100%) !important;
}

/* ── EXPANDERS ───────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 500;
    color: #475569;
    padding: 12px 16px;
}

/* ── RISK BADGES ─────────────────────────────────────── */
.badge-critical {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    color: #991b1b;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    border: 1px solid #fecaca;
}
.badge-high {
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    color: #9a3412;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    border: 1px solid #fed7aa;
}
.badge-medium {
    background: linear-gradient(135deg, #fefce8, #fef9c3);
    color: #854d0e;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    border: 1px solid #fde68a;
}
.badge-low {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    color: #166534;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    border: 1px solid #bbf7d0;
}

/* ── SPINNER ─────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #6366f1;
}

/* ── PLOTLY CHARTS ───────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}
</style>
""",
    unsafe_allow_html=True,
)


# ── CACHED RESOURCE / DATA LOADERS ───────────────────────────
@st.cache_resource
def get_model_scorer() -> CustomerRiskScorer:
    """Load and cache the trained ML model ensemble scorer."""
    return CustomerRiskScorer()


@st.cache_data
def get_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load precomputed demo dataset and risk profiles."""
    df = load_clean_data()
    risk_df = load_risk_profiles()
    return df, risk_df


@st.cache_data
def get_demo_reports() -> tuple[str, dict[str, str]]:
    """Load saved executive briefing and segment recommendations."""
    return load_saved_briefing_reports()


# Initialize scorer and demo datasets
scorer = get_model_scorer()
demo_df, demo_risk_df = get_demo_data()
demo_briefing, demo_recs = get_demo_reports()

# Determine active dataset mode (Demo vs Upload)
is_uploaded = (
    st.session_state.get("data_mode") == "upload"
    and "uploaded_df" in st.session_state
    and "uploaded_risk_df" in st.session_state
)

if is_uploaded:
    df = st.session_state["uploaded_df"]
    risk_df = st.session_state["uploaded_risk_df"]
else:
    df = demo_df
    risk_df = demo_risk_df

# Compute unified SaaS KPIs for active dataset
kpis = compute_saas_kpis(df)
health = compute_business_health_score(
    kpis["churn_rate"], kpis["rev_retention"], kpis["ltv_cac"]
)

# Segment risk counts
risk_counts = {
    level: int((risk_df["risk_level"] == level).sum())
    for level in ["Critical", "High", "Medium", "Low"]
}

# ── SIDEBAR NAVIGATION ───────────────────────────────────────
st.sidebar.title("SaaS Growth Intelligence")
st.sidebar.caption("Customer churn analysis & retention dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Data Upload & Scoring",
        "Executive Dashboard",
        "Customer Risk Explorer",
        "Segment Analysis",
        "Revenue Calculator",
        "Retention Playbook",
    ],
)

st.sidebar.markdown("---")
mode_label = "🟢 Uploaded Dataset" if is_uploaded else "🔵 Demo Dataset (Telco)"
st.sidebar.markdown(f"**Active Mode:** {mode_label}")
st.sidebar.markdown(f"**Total Records:** {kpis['total_customers']:,}")
st.sidebar.markdown(f"**Churn Rate:** {kpis['churn_rate']}%")
st.sidebar.markdown(f"**Critical Alerts:** {risk_counts['Critical']:,} accounts")
st.sidebar.markdown(f"**Annual Rev at Risk:** ${kpis['churn_arr']/1e6:.2f}M")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built by Niviya Albert · Python, Scikit-Learn, Streamlit, Gemini AI"
)


# ═════════════════════════════════════════════════════════════
# PAGE 0 — DATA UPLOAD & DYNAMIC SCORING
# ═════════════════════════════════════════════════════════════
if page == "Data Upload & Scoring":
    st.title("Data Upload & Customer Risk Scoring")
    st.markdown(
        "Upload a customer CSV to see which accounts are at risk, why, and what to do about it."
    )
    st.markdown("---")

    # ── STATUS BANNER ─────────────────────────────────────────
    if is_uploaded:
        st.success(
            f"🟢 **Active Dataset:** Uploaded — {kpis['total_customers']:,} accounts scored | "
            f"Churn Rate: **{kpis['churn_rate']}%** | "
            f"Critical ARR Exposure: **${kpis['churn_arr']/1e6:.2f}M**"
        )
    else:
        st.info("🔵 **Active Dataset:** Built-in Telco Demo (7,032 accounts) — Upload your data below or keep exploring the demo.")

    st.markdown("---")

    # ── OPTION CARDS ──────────────────────────────────────────
    col_opt1, col_opt2 = st.columns(2, gap="large")

    with col_opt1:
        st.markdown(
            """
            <div class="report-card">
            <h4>🔵 &nbsp;Option 1 — Built-in Telco Demo</h4>
            <p>Explore a full analysis of <strong>7,032 real telecom customers</strong> — retention trends,
            churn drivers, revenue impact, and an AI-written executive summary.</p>
            <ul>
              <li>✅ No file upload needed — works immediately</li>
              <li>✅ All 6 dashboard pages enabled</li>
              <li>✅ Executive briefing pre-generated</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("▶  Load Telco Demo Dataset", use_container_width=True):
            st.session_state["data_mode"] = "demo"
            st.session_state.pop("uploaded_df", None)
            st.session_state.pop("uploaded_risk_df", None)
            st.success("✅ Demo dataset active. Navigate to Executive Dashboard!")
            st.rerun()

    with col_opt2:
        st.markdown(
            """
            <div class="report-card">
            <h4>🟢 &nbsp;Option 2 — Upload Your Customer CSV</h4>
            <p>Upload your own customer CSV and get <strong>individual churn risk scores
            and revenue exposure</strong> calculated for every account automatically.</p>
            <ul>
              <li>✅ Automatic data validation & cleaning</li>
              <li>✅ What-If retention scenario simulator</li>
              <li>✅ Full revenue exposure breakdown by risk tier</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── UPLOAD ZONE ───────────────────────────────────────────
    st.subheader("📂 Upload Customer Data")

    upload_col, schema_col = st.columns([3, 2], gap="large")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Drop your CSV here or click to browse",
            type=["csv"],
            help="Accepts CSV files up to 200MB. Required columns: customerID, tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, Churn.",
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.caption("📋 **Required columns:** `customerID` · `tenure` · `MonthlyCharges` · `TotalCharges` · `Contract` · `PaymentMethod` · `Churn`")

    with schema_col:
        with st.expander("📖 View Full Column Schema", expanded=False):
            sample_spec = pd.DataFrame(
                {
                    "Column": [
                        "customerID", "tenure", "MonthlyCharges",
                        "TotalCharges", "Contract", "PaymentMethod",
                        "Churn", "InternetService", "TechSupport",
                    ],
                    "Type": [
                        "String", "Integer (months)", "Float ($)",
                        "Float ($)", "Month-to-month / One year / Two year",
                        "String", "Yes/No or 1/0",
                        "String (Optional)", "Yes/No (Optional)",
                    ],
                    "Example": [
                        "CUST-8491", "14", "75.50",
                        "1057.00", "Month-to-month",
                        "Electronic check", "Yes",
                        "Fiber optic", "No",
                    ],
                }
            )
            st.dataframe(sample_spec, use_container_width=True, hide_index=True)

    # ── FILE PROCESSING ───────────────────────────────────────
    if uploaded_file is not None:
        try:
            raw_input_df = pd.read_csv(uploaded_file)

            # Preview
            st.markdown("#### Raw Data Preview")
            st.dataframe(raw_input_df.head(5), use_container_width=True)
            st.caption(f"Loaded **{len(raw_input_df):,} rows × {len(raw_input_df.columns)} columns** from `{uploaded_file.name}`")

            st.markdown("---")

            # Processing pipeline
            with st.spinner("🔄 Validating and scoring your customers..."):
                cleaned_user_df, errors, dropped = clean_customer_data(raw_input_df, min_rows=5)

                if errors:
                    st.error("⚠️ **Validation Failed:** " + " | ".join(errors))
                else:
                    scored_user_df = scorer.score_dataframe(cleaned_user_df)

                    st.session_state["uploaded_df"] = cleaned_user_df
                    st.session_state["uploaded_risk_df"] = scored_user_df
                    st.session_state["data_mode"] = "upload"

            if not errors:
                # Success header
                st.success(
                    f"✅ **Done!** {len(cleaned_user_df):,} customers scored "
                    f"({dropped} invalid rows removed)."
                )

                # Results summary metrics
                st.markdown("#### 📊 Results Summary")
                u_kpis = compute_saas_kpis(cleaned_user_df)
                u_crit = int((scored_user_df["risk_level"] == "Critical").sum())
                u_high = int((scored_user_df["risk_level"] == "High").sum())
                u_crit_arr = scored_user_df[scored_user_df["risk_level"] == "Critical"]["annual_revenue_risk"].sum()

                q1, q2, q3, q4, q5 = st.columns(5)
                q1.metric("📋 Accounts Scored", f"{u_kpis['total_customers']:,}")
                q2.metric("📉 Churn Rate", f"{u_kpis['churn_rate']}%")
                q3.metric("🔴 Critical Risk", f"{u_crit:,}")
                q4.metric("🟠 High Risk", f"{u_high:,}")
                q5.metric("💰 Critical ARR Exposure", f"${u_crit_arr:,.0f}")

                st.markdown("---")

                # Risk tier breakdown table
                tier_summary = (
                    scored_user_df.groupby("risk_level")
                    .agg(
                        Accounts=("customerID", "count"),
                        Avg_Score=("risk_score", "mean"),
                        ARR_Exposure=("annual_revenue_risk", "sum"),
                    )
                    .reset_index()
                    .rename(columns={"risk_level": "Risk Tier"})
                )
                tier_summary["Avg_Score"] = tier_summary["Avg_Score"].round(1)
                tier_summary["ARR_Exposure"] = tier_summary["ARR_Exposure"].apply(lambda x: f"${x:,.0f}")
                tier_summary.columns = ["Risk Tier", "Accounts", "Avg Risk Score", "Annual ARR Exposure"]
                st.dataframe(tier_summary, use_container_width=True, hide_index=True)

                st.info("👉 **Your dataset is now active.** Navigate to any page — all analysis, What-If modeling, and revenue calculations now use your data.")

        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Error processing CSV: {e!s}")


# ═════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ═════════════════════════════════════════════════════════════
elif page == "Executive Dashboard":
    st.title("Executive Retention Intelligence")
    st.markdown(
        f"Real-time revenue risk visibility across **{kpis['total_customers']:,} active accounts**."
    )
    st.markdown("---")

    # ── ROW 1: PRIMARY FINANCIAL METRICS ─────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Annual Churn Rate",
            f"{kpis['churn_rate']}%",
            delta=f"{round(kpis['churn_rate'] - 5.0, 2)}% vs 5% benchmark",
            delta_color="inverse",
        )
    with m2:
        st.metric(
            "Critical Risk Accounts",
            f"{risk_counts['Critical']:,}",
            delta="Urgent intervention needed",
            delta_color="inverse",
        )
    with m3:
        st.metric(
            "Annual Revenue at Risk",
            f"${kpis['churn_arr']/1e6:.2f}M",
            delta=f"${kpis['churn_mrr']:,.0f}/month",
            delta_color="inverse",
        )
    with m4:
        delta_type = "normal" if kpis["ltv_cac"] >= 3.0 else "inverse"
        st.metric(
            "LTV : CAC Ratio",
            f"{kpis['ltv_cac']}:1",
            delta="Target > 3.0:1",
            delta_color=delta_type,
        )

    st.markdown("---")

    # ── ROW 2: BUSINESS HEALTH SCORE & AUDIT ──────────────────
    h_col1, h_col2 = st.columns([1, 1.5])

    with h_col1:
        st.subheader("Business Health Score")
        fig_health = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health["health_score"],
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": health["colour"]},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(231,76,60,0.12)"},
                        {"range": [50, 70], "color": "rgba(230,126,34,0.12)"},
                        {"range": [70, 100], "color": "rgba(39,174,96,0.12)"},
                    ],
                },
                title={"text": f"Status: {health['status']}"},
            )
        )
        fig_health.update_layout(height=230, margin={"t": 40, "b": 10, "l": 15, "r": 15})
        st.plotly_chart(fig_health, use_container_width=True)

    with h_col2:
        st.subheader("Health Component Breakdown")
        health_table = pd.DataFrame(health["breakdown"])
        st.dataframe(health_table, use_container_width=True, hide_index=True)
        st.caption(
            f"Overall health standing: **{health['health_score']}/100** ({health['status']}). "
            f"Net Revenue Retention is currently **{kpis['rev_retention']}%**."
        )

    st.markdown("---")

    # ── ROW 3: AI EXECUTIVE BRIEFING ─────────────────────────
    st.subheader("🤖 AI-Generated Executive Briefing")

    # Interactive live generation option
    with st.expander("⚡ Live Gemini AI Briefing Generator (Optional)", expanded=False):
        gemini_key = st.text_input(
            "Enter Google Gemini API Key (or leave blank to use server environment):",
            type="password",
            placeholder="AIzaSy...",
        )
        if st.button("Generate Fresh AI Briefing"):
            with st.spinner("Synthesizing metrics and querying Gemini 2.5 Flash..."):
                metrics_sum = build_metrics_summary(df, risk_df)
                new_briefing, new_recs = generate_executive_briefing(
                    metrics_sum, api_key=gemini_key if gemini_key else None
                )
                st.session_state["live_briefing"] = new_briefing
                st.session_state["live_recs"] = new_recs
                st.success("Executive briefing generated successfully!")

    active_briefing = st.session_state.get("live_briefing", demo_briefing)

    if active_briefing:
        st.markdown(
            f"""
<div class="report-card">
{active_briefing}
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Generate or load executive briefing above.")

    st.markdown("---")

    # ── ROW 4: RISK DISTRIBUTION & REVENUE EXPOSURE ──────────
    st.subheader("Portfolio Risk Distribution")
    r_col1, r_col2 = st.columns(2)

    levels = ["Critical", "High", "Medium", "Low"]
    counts = [risk_counts[lvl] for lvl in levels]
    colours = ["#e74c3c", "#e67e22", "#f1c40f", "#27ae60"]

    with r_col1:
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=levels,
                    values=counts,
                    hole=0.48,
                    marker={"colors": colours},
                    textinfo="label+percent",
                )
            ]
        )
        fig_donut.update_layout(
            title="Accounts by Risk Tier", height=320, margin={"t": 40, "b": 10, "l": 10, "r": 10}
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with r_col2:
        rev_by_level = [
            round(risk_df[risk_df["risk_level"] == lvl]["annual_revenue_risk"].sum() / 1e3, 1)
            for lvl in levels
        ]
        fig_rev = go.Figure(
            data=[
                go.Bar(
                    x=levels,
                    y=rev_by_level,
                    marker_color=colours,
                    text=[f"${v:,.0f}K" for v in rev_by_level],
                    textposition="outside",
                )
            ]
        )
        fig_rev.update_layout(
            title="Annual ARR Exposure by Risk Tier ($ Thousands)",
            yaxis_title="ARR at Risk ($K)",
            height=320,
            margin={"t": 40, "b": 10},
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # ── ROW 5: TOP 10 PRIORITY CUSTOMERS ─────────────────────
    st.subheader("🚨 Top 10 Priority Accounts Requiring Immediate Contact")
    top10_df = risk_df.head(10)[
        [
            "priority_rank",
            "customerID",
            "risk_score",
            "risk_level",
            "monthly_revenue",
            "tenure_months",
            "contract_type",
            "churn_drivers",
        ]
    ].copy()
    top10_df.columns = [
        "Rank",
        "Customer ID",
        "Risk Score (0-100)",
        "Risk Tier",
        "Monthly Value ($)",
        "Tenure (Mo)",
        "Contract Type",
        "Top Churn Drivers",
    ]
    st.dataframe(top10_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER RISK EXPLORER & WHAT-IF SIMULATOR
# ═════════════════════════════════════════════════════════════
elif page == "Customer Risk Explorer":
    st.title("Customer Risk Explorer & Dynamic Simulator")
    st.markdown(
        "Inspect individual account profiles with **live machine learning feature attributions** "
        "and simulate retention intervention scenarios in real time."
    )
    st.markdown("---")

    search_col, select_col = st.columns(2)
    with search_col:
        search_id = st.text_input(
            "Search Customer ID", placeholder="e.g. 9300-AGZNL"
        ).strip().upper()

    with select_col:
        top_critical_ids = risk_df.head(25)["customerID"].tolist()
        selected_id = st.selectbox("Or choose from top critical accounts:", ["-- Select --"] + top_critical_ids)
        if selected_id != "-- Select --":
            search_id = selected_id

    if search_id:
        match_rows = risk_df[risk_df["customerID"].astype(str).str.upper() == search_id]
        if len(match_rows) == 0:
            st.error(f"Customer ID '{search_id}' not found in active dataset.")
        else:
            cust = match_rows.iloc[0]
            raw_match = df[df["customerID"].astype(str).str.upper() == search_id]
            raw_cust_dict = raw_match.iloc[0].to_dict() if len(raw_match) > 0 else cust.to_dict()

            # Profile Header Metrics
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Risk Score", f"{cust['risk_score']}/100")
            p2.metric("Risk Level", cust["risk_level"])
            p3.metric("Ensemble Churn Probability", f"{round(cust['ensemble_probability']*100, 1)}%")
            p4.metric("Priority Rank", f"#{int(cust['priority_rank']):,} of {len(risk_df):,}")

            st.markdown("---")

            col_details, col_drivers = st.columns(2)
            with col_details:
                st.subheader("Account Metadata")
                outcome_text = (
                    "🔴 Churned"
                    if cust.get("actual_churn") == 1
                    else "🟢 Active Account"
                )
                st.markdown(
                    f"**Customer ID:** `{cust['customerID']}`  \n"
                    f"**Tenure:** {int(cust['tenure_months'])} months  \n"
                    f"**Contract:** {cust['contract_type']}  \n"
                    f"**Payment Method:** {cust['payment_method']}  \n"
                    f"**Monthly Billing:** ${cust['monthly_revenue']:.2f}/mo  \n"
                    f"**Annual Revenue Exposure:** ${cust['annual_revenue_risk']:,.2f}  \n"
                    f"**Recorded Outcome:** {outcome_text}"
                )

            with col_drivers:
                st.subheader("Top Churn Drivers & Playbook")
                drivers = cust["churn_drivers"].split(" | ")
                for i, d in enumerate(drivers, 1):
                    st.markdown(f"**{i}.** {d}")

                st.markdown("**Recommended Prescriptive Action:**")
                st.info(cust["recommended_action"])

            st.markdown("---")

            # ── LIVE ML WHAT-IF SIMULATOR ────────────────────
            st.subheader("⚡ Real-Time ML What-If Simulator")
            st.markdown(
                "Modify contract terms or billing methods below. The input vector is passed directly "
                "into the **trained Gradient Boosting + Random Forest + Logistic Regression ensemble**."
            )

            sim_c1, sim_c2 = st.columns(2)

            contract_options = ["Month-to-month", "One year", "Two year"]
            cur_contract = cust["contract_type"]
            c_idx = (
                contract_options.index(cur_contract)
                if cur_contract in contract_options
                else 0
            )

            with sim_c1:
                new_contract = st.selectbox(
                    "Simulate Contract Upgrade:", contract_options, index=c_idx
                )

            payment_options = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
            cur_payment = cust["payment_method"]
            p_idx = (
                payment_options.index(cur_payment)
                if cur_payment in payment_options
                else 0
            )

            with sim_c2:
                new_payment = st.selectbox(
                    "Simulate Payment Method Change:", payment_options, index=p_idx
                )

            # Execute real model inference simulation
            sim_result = scorer.simulate_what_if(
                raw_cust_dict, new_contract=new_contract, new_payment=new_payment
            )

            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric(
                "Simulated Risk Score",
                f"{sim_result['new_score']}/100",
                delta=f"{sim_result['score_delta']} pts",
                delta_color="inverse",
            )
            res_c2.metric(
                "Simulated Churn Probability",
                f"{round(sim_result['new_probability']*100, 1)}%",
                delta=f"{round((sim_result['new_probability'] - sim_result['base_probability'])*100, 1)}%",
                delta_color="inverse",
            )
            res_c3.metric("Simulated Risk Tier", sim_result["new_risk_level"])

            if sim_result["score_delta"] < 0:
                st.success(
                    f"🎉 **Retention Impact:** Risk score drops by **{abs(sim_result['score_delta'])} points**! "
                    f"New probability: **{round(sim_result['new_probability']*100, 1)}%**."
                )
            elif sim_result["score_delta"] == 0:
                st.info("No change from baseline parameters.")


# ═════════════════════════════════════════════════════════════
# PAGE 3 — SEGMENT ANALYSIS
# ═════════════════════════════════════════════════════════════
elif page == "Segment Analysis":
    st.title("Customer Segment & Cohort Intelligence")
    st.markdown("Explore churn risk variations across contract types, payment channels, and tenure cohorts.")
    st.markdown("---")

    f1, f2, f3 = st.columns(3)
    with f1:
        contracts = st.multiselect(
            "Filter Contract Type",
            options=df["Contract"].unique().tolist(),
            default=df["Contract"].unique().tolist(),
        )
    with f2:
        payments = st.multiselect(
            "Filter Payment Method",
            options=df["PaymentMethod"].unique().tolist(),
            default=df["PaymentMethod"].unique().tolist(),
        )
    with f3:
        max_t = int(df["tenure"].max()) if len(df) > 0 else 72
        tenure_range = st.slider("Filter Tenure Range (Months)", 0, max_t, (0, max_t))

    filter_mask = (
        df["Contract"].isin(contracts)
        & df["PaymentMethod"].isin(payments)
        & df["tenure"].between(tenure_range[0], tenure_range[1])
    )
    filt_df = df[filter_mask].copy()

    if len(filt_df) == 0:
        st.warning("No customer records match the selected filter criteria.")
    else:
        f_kpis = compute_saas_kpis(filt_df)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Filtered Accounts", f"{f_kpis['total_customers']:,}")
        s2.metric(
            "Segment Churn Rate",
            f"{f_kpis['churn_rate']}%",
            delta=f"{round(f_kpis['churn_rate'] - kpis['churn_rate'], 2)}% vs avg",
            delta_color="inverse",
        )
        s3.metric("Churned Accounts", f"{f_kpis['total_churned']:,}")
        s4.metric("Segment ARR Exposure", f"${f_kpis['churn_arr']:,.0f}")

        st.markdown("---")

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.subheader("Churn Rate by Contract Type")
            ct_grp = filt_df.groupby("Contract")["Churn"].mean().mul(100).round(1).reset_index()
            fig_ct = px.bar(
                ct_grp,
                x="Contract",
                y="Churn",
                color="Churn",
                color_continuous_scale=["#27ae60", "#f1c40f", "#e74c3c"],
                text="Churn",
            )
            fig_ct.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_ct.update_layout(showlegend=False, height=340, yaxis_title="Churn Rate (%)")
            st.plotly_chart(fig_ct, use_container_width=True)

        with c_chart2:
            st.subheader("Churn Rate by Payment Method")
            pm_grp = filt_df.groupby("PaymentMethod")["Churn"].mean().mul(100).round(1).reset_index()
            pm_grp = pm_grp.sort_values(by="Churn")
            fig_pm = px.bar(
                pm_grp,
                x="Churn",
                y="PaymentMethod",
                orientation="h",
                color="Churn",
                color_continuous_scale=["#27ae60", "#f1c40f", "#e74c3c"],
                text="Churn",
            )
            fig_pm.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_pm.update_layout(showlegend=False, height=340, xaxis_title="Churn Rate (%)")
            st.plotly_chart(fig_pm, use_container_width=True)

        st.markdown("---")

        # Tenure Progression
        st.subheader("Churn Rate by Lifecycle Tenure Bucket")
        filt_df["tenure_bucket"] = pd.cut(
            filt_df["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 100],
            labels=["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61m+"],
        )
        tb_grp = filt_df.groupby("tenure_bucket", observed=True)["Churn"].mean().mul(100).round(1).reset_index()
        fig_tb = px.line(
            tb_grp,
            x="tenure_bucket",
            y="Churn",
            markers=True,
            text="Churn",
            color_discrete_sequence=["#e74c3c"],
        )
        fig_tb.update_traces(textposition="top center", texttemplate="%{text}%")
        fig_tb.update_layout(height=320, yaxis_title="Churn Rate (%)", xaxis_title="Tenure Bucket")
        st.plotly_chart(fig_tb, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 4 — REVENUE CALCULATOR & ROI MODELER
# ═════════════════════════════════════════════════════════════
elif page == "Revenue Calculator":
    st.title("Retention Revenue Impact & ROI Modeler")
    st.markdown(
        "Build executive business cases for customer success investments with scenario modeling "
        "and payback period analysis."
    )
    st.markdown("---")

    target_pct = st.slider(
        "Target Churn Reduction (%)",
        min_value=5,
        max_value=75,
        value=20,
        step=5,
        help="Percentage of currently churned revenue you aim to retain.",
    )

    customers_saved = round(kpis["total_churned"] * (target_pct / 100.0))
    revenue_saved = round(customers_saved * kpis["avg_monthly"] * 12.0, 0)
    new_churn_rate = round(kpis["churn_rate"] * (1.0 - target_pct / 100.0), 2)
    new_mrr = round(kpis["mrr"] + (customers_saved * kpis["avg_monthly"]), 0)

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Accounts Retained", f"{customers_saved:,}", f"of {kpis['total_churned']:,} churners")
    rc2.metric("Annual Revenue Saved", f"${revenue_saved:,.0f}", delta=f"${revenue_saved/12:,.0f}/mo", delta_color="normal")
    rc3.metric("Projected Churn Rate", f"{new_churn_rate}%", delta=f"-{round(kpis['churn_rate'] - new_churn_rate, 2)}% drop", delta_color="normal")
    rc4.metric("New Projected MRR", f"${new_mrr:,.0f}", delta=f"+${customers_saved*kpis['avg_monthly']:,.0f}/mo", delta_color="normal")

    st.markdown("---")

    st.subheader("Financial Return on Retention Investment (ROI)")
    programme_cost = st.number_input(
        "Annual Retention / Customer Success Budget ($):",
        min_value=0,
        max_value=1000000,
        value=50000,
        step=5000,
    )

    roi_stats = calculate_retention_roi(revenue_saved, programme_cost)

    roi_c1, roi_c2, roi_c3 = st.columns(3)
    roi_c1.metric(
        "Net Annual Revenue Gain",
        f"${roi_stats['net_gain']:,.0f}",
        delta=f"After ${programme_cost:,} budget",
        delta_color="normal" if roi_stats["net_gain"] > 0 else "inverse",
    )
    roi_c2.metric("ROI Multiple", f"{roi_stats['roi_pct']}%", delta_color="normal")
    roi_c3.metric("Payback Period", f"{roi_stats['payback_months']} Months", delta_color="normal")

    st.markdown("---")

    st.subheader("Sensitivity Matrix Across Reduction Targets")
    scen_df = calculate_reduction_scenarios(
        kpis["total_churned"], kpis["avg_monthly"], kpis["churn_rate"], kpis["mrr"]
    )
    fig_scen = px.bar(
        scen_df,
        x="Reduction %",
        y="Revenue Saved",
        color="Revenue Saved",
        color_continuous_scale=["#f1c40f", "#27ae60"],
        text="Revenue Saved",
    )
    fig_scen.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", textfont_size=10)
    fig_scen.add_vline(x=target_pct, line_dash="dash", line_color="red", annotation_text="Selected Target")
    fig_scen.update_layout(showlegend=False, height=360, yaxis_title="Annual ARR Saved ($)")
    st.plotly_chart(fig_scen, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 5 — RETENTION PLAYBOOK & ACTION LIST
# ═════════════════════════════════════════════════════════════
elif page == "Retention Playbook":
    st.title("Prescriptive Retention Playbook & Action Lists")
    st.markdown(
        "Operational retention strategies and priority account contact lists ready for CSM and sales outreach."
    )
    st.markdown("---")

    active_recs = st.session_state.get("live_recs", demo_recs)

    st.subheader("Strategic Playbooks by Risk Segment")
    for level in ["Critical", "High", "Medium", "Low"]:
        cnt = risk_counts.get(level, 0)
        rev_exp = risk_df[risk_df["risk_level"] == level]["annual_revenue_risk"].sum()
        badge_class = f"badge-{level.lower()}"

        with st.expander(f"{level} Risk Segment — {cnt:,} Accounts | ${rev_exp:,.0f} ARR Exposure", expanded=(level == "Critical")):
            st.markdown(active_recs.get(level, "Standard retention playbook applied."))

    st.markdown("---")

    st.subheader("Exportable Account Outreach Queue")
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        sel_tiers = st.multiselect(
            "Filter Risk Tiers:",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with f_c2:
        row_limit = st.slider("Max Accounts to Display:", 10, 500, 100)

    if sel_tiers:
        queue_df = risk_df[risk_df["risk_level"].isin(sel_tiers)].head(row_limit)[
            [
                "priority_rank",
                "customerID",
                "risk_score",
                "risk_level",
                "monthly_revenue",
                "tenure_months",
                "contract_type",
                "churn_drivers",
                "recommended_action",
            ]
        ].copy()

        queue_df.columns = [
            "Rank",
            "Customer ID",
            "Score",
            "Tier",
            "Monthly ($)",
            "Tenure",
            "Contract",
            "Churn Drivers",
            "Prescriptive Playbook",
        ]

        st.dataframe(queue_df, use_container_width=True, hide_index=True)

        csv_data = queue_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Action Queue (CSV)",
            data=csv_data,
            file_name="saas_retention_action_queue.csv",
            mime="text/csv",
        )
    else:
        st.info("Select at least one risk tier above to generate the outreach queue.")
