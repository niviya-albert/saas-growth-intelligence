import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pathlib
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Growth Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: rgba(30,30,46,0.05);
    border: 1px solid rgba(45,45,63,0.2);
    border-radius: 12px;
    padding: 16px;
}
h1 { font-weight: 600; }
h2 { font-weight: 500; }
.stAlert { border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 8px; }
.stDownloadButton button {
    background-color: #27ae60;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── PATHS ─────────────────────────────────────────────────
BASE    = pathlib.Path(__file__).parent.parent
DATA    = BASE / 'data' / 'cleaned'
REPORTS = BASE / 'reports'

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df      = pd.read_csv(DATA / 'telco_churn_clean.csv')
    risk_df = pd.read_csv(DATA / 'customer_risk_profiles.csv')
    return df, risk_df

@st.cache_data
def load_reports():
    briefing      = ""
    recs          = {}
    briefing_path = REPORTS / 'executive_briefing.txt'
    recs_path     = REPORTS / 'segment_recommendations.json'
    if briefing_path.exists():
        briefing = briefing_path.read_text(encoding='utf-8')
    if recs_path.exists():
        recs = json.loads(
            recs_path.read_text(encoding='utf-8')
        )
    return briefing, recs

df, risk_df      = load_data()
briefing, recs   = load_reports()

# ── PRECOMPUTE METRICS ────────────────────────────────────
total_customers = len(df)
churn_rate      = round(df['Churn'].mean() * 100, 2)
mrr             = round(
    df[df['Churn']==0]['MonthlyCharges'].sum(), 0
)
churn_mrr       = round(
    df[df['Churn']==1]['MonthlyCharges'].sum(), 0
)
churn_arr       = round(churn_mrr * 12, 0)
avg_monthly     = round(df['MonthlyCharges'].mean(), 2)
avg_tenure      = round(
    df[df['Churn']==1]['tenure'].mean(), 1
)
ltv             = round(avg_monthly * avg_tenure, 2)
ltv_cac         = round(ltv / 300, 2)
rev_retention   = round(
    mrr / (mrr + churn_mrr) * 100, 1
)
total_churned   = int(df['Churn'].sum())

risk_counts = {
    level: int((risk_df['risk_level'] == level).sum())
    for level in ['Critical', 'High', 'Medium', 'Low']
}
critical_rev = round(
    risk_df[
        risk_df['risk_level']=='Critical'
    ]['annual_revenue_risk'].sum(), 0
)

# ── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("📊 SaaS Growth\nIntelligence System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Data Upload",
        "Executive Dashboard",
        "Customer Risk Explorer",
        "Segment Analysis",
        "Revenue Calculator",
        "Recommendation Engine"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset:** {total_customers:,} customers  \n"
    f"**Churn Rate:** {churn_rate}%  \n"
    f"**Critical Alerts:** {risk_counts['Critical']:,}  \n"
    f"**Annual Risk:** ${churn_arr/1e6:.2f}M"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with Python · Scikit-learn · Gemini AI  \n"
    "Portfolio Project · Data Analytics 2026"
)


# ══════════════════════════════════════════════════════════
# PAGE 0 — DATA UPLOAD
# ══════════════════════════════════════════════════════════
if page == "Data Upload":

    st.title("Data Upload")
    st.markdown(
        "Upload your own customer data to generate "
        "a personalised retention intelligence report. "
        "Or use the built-in Telco demo dataset."
    )
    st.markdown("---")

    # ── DEMO MODE ─────────────────────────────────────────
    st.subheader("Option 1 — Use Demo Dataset")
    st.markdown(
        "Explore the system using 7,032 real SaaS "
        "customers from a telecom company. All analysis "
        "is pre-built and ready to explore."
    )

    if st.button("Load Demo Dataset"):
        st.session_state['data_mode'] = 'demo'
        st.success(
            "Demo dataset loaded. "
            "Navigate to Executive Dashboard to begin."
        )

    st.markdown("---")

    # ── UPLOAD MODE ───────────────────────────────────────
    st.subheader("Option 2 — Upload Your Own Data")
    st.markdown(
        "Upload a CSV file with your customer data. "
        "The system will analyse it automatically."
    )

    # Required columns explainer
    with st.expander(
        "Required CSV format — click to see",
        expanded=False
    ):
        st.markdown(
            "Your CSV must contain these columns "
            "(column names must match exactly):"
        )

        required_cols = pd.DataFrame({
            'Column': [
                'customerID', 'tenure',
                'MonthlyCharges', 'TotalCharges',
                'Contract', 'PaymentMethod',
                'Churn', 'InternetService',
                'OnlineSecurity', 'TechSupport'
            ],
            'Type': [
                'Text', 'Number (months)',
                'Number ($)', 'Number ($)',
                'Text', 'Text',
                'Yes/No or 1/0', 'Text',
                'Yes/No', 'Yes/No'
            ],
            'Example': [
                '7590-VHVEG', '24',
                '65.50', '1572.00',
                'Month-to-month', 'Electronic check',
                'Yes', 'Fiber optic',
                'No', 'Yes'
            ]
        })
        st.dataframe(
            required_cols,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "**Churn column:** accepts Yes/No text "
            "or 1/0 numbers.  \n"
            "**Contract column:** accepts "
            "Month-to-month, One year, Two year.  \n"
            "**Minimum rows:** 100 customers "
            "for reliable analysis."
        )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your customer CSV file",
        type=['csv'],
        help="Maximum file size: 200MB"
    )

    if uploaded_file is not None:
        try:
            # Load uploaded data
            user_df = pd.read_csv(uploaded_file)

            st.markdown("---")
            st.subheader("Data Preview")

            # Show shape
            st.markdown(
                f"**Rows:** {len(user_df):,}  |  "
                f"**Columns:** {user_df.shape[1]}"
            )

            # Show first 5 rows
            st.dataframe(
                user_df.head(),
                use_container_width=True,
                hide_index=True
            )

            # ── VALIDATE REQUIRED COLUMNS ─────────────────
            required = [
                'customerID', 'tenure',
                'MonthlyCharges', 'TotalCharges',
                'Contract', 'PaymentMethod', 'Churn'
            ]
            missing_cols = [
                c for c in required
                if c not in user_df.columns
            ]

            if missing_cols:
                st.error(
                    f"Missing required columns: "
                    f"{', '.join(missing_cols)}  \n"
                    f"Please check the format guide above "
                    f"and re-upload."
                )
            else:
                # ── BASIC CLEANING ────────────────────────
                # Handle Yes/No Churn column
                if user_df['Churn'].dtype == object:
                    user_df['Churn'] = user_df[
                        'Churn'
                    ].map({'Yes': 1, 'No': 0})

                # Convert TotalCharges to numeric
                user_df['TotalCharges'] = pd.to_numeric(
                    user_df['TotalCharges'],
                    errors='coerce'
                )

                # Drop incomplete rows
                user_df.dropna(
                    subset=['TotalCharges', 'Churn'],
                    inplace=True
                )

                # Calculate basic metrics
                u_churn = round(
                    user_df['Churn'].mean() * 100, 2
                )
                u_mrr   = round(
                    user_df[
                        user_df['Churn']==0
                    ]['MonthlyCharges'].sum(), 0
                )
                u_churn_mrr = round(
                    user_df[
                        user_df['Churn']==1
                    ]['MonthlyCharges'].sum(), 0
                )
                u_churn_arr = round(u_churn_mrr * 12, 0)

                # Show quick summary
                st.markdown("---")
                st.subheader("Quick Analysis")

                q1, q2, q3, q4 = st.columns(4)
                q1.metric(
                    "Total Customers",
                    f"{len(user_df):,}"
                )
                q2.metric(
                    "Churn Rate",
                    f"{u_churn}%"
                )
                q3.metric(
                    "Monthly Revenue",
                    f"${u_mrr:,.0f}"
                )
                q4.metric(
                    "Annual Churn Cost",
                    f"${u_churn_arr:,.0f}"
                )

                # Contract churn breakdown
                if 'Contract' in user_df.columns:
                    st.markdown("**Churn by Contract:**")
                    ct = user_df.groupby('Contract')[
                        'Churn'
                    ].mean().mul(100).round(1).reset_index()
                    ct.columns = [
                        'Contract', 'Churn Rate (%)'
                    ]
                    fig_ct = px.bar(
                        ct,
                        x='Contract',
                        y='Churn Rate (%)',
                        color='Churn Rate (%)',
                        color_continuous_scale=[
                            '#27ae60', '#f1c40f', '#e74c3c'
                        ],
                        text='Churn Rate (%)'
                    )
                    fig_ct.update_traces(
                        texttemplate='%{text}%',
                        textposition='outside'
                    )
                    fig_ct.update_layout(
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(
                        fig_ct,
                        use_container_width=True
                    )

                # Save to session state
                # so other pages can use it
                st.session_state['uploaded_df'] = user_df
                st.session_state['data_mode']   = 'upload'

                st.success(
                    f"Data validated successfully. "
                    f"{len(user_df):,} customers loaded.  \n"
                    f"Navigate to any page to explore "
                    f"your data."
                )

                # Note about full analysis
                st.info(
                    "**Note:** The full risk scoring, "
                    "ML predictions, and AI briefing "
                    "on other pages use the pre-built "
                    "demo models. For a fully customised "
                    "analysis on your data, contact me "
                    "to discuss a tailored engagement."
                )

        except Exception as e:
            st.error(
                f"Error reading file: {str(e)}  \n"
                f"Make sure your file is a valid CSV."
            )

    st.markdown("---")
    st.markdown(
        "**Questions about the format?**  \n"
        "The system works best with at least 500 "
        "customer records and 12+ months of history.  \n"
        "For custom implementations on proprietary "
        "data formats, contact me directly."
    )

# ══════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════
if page == "Executive Dashboard":

    st.title("Executive Dashboard")
    st.markdown(
        "Real-time customer retention intelligence "
        f"across {total_customers:,} active customers."
    )
    st.markdown("---")

    # ── ROW 1: KEY METRICS ────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="Annual Churn Rate",
            value=f"{churn_rate}%",
            delta=f"{round(churn_rate - 5.0, 2)}% above benchmark",
            delta_color="inverse"
        )
    with c2:
        st.metric(
            label="Critical Risk Customers",
            value=f"{risk_counts['Critical']:,}",
            delta="Require immediate action",
            delta_color="inverse"
        )
    with c3:
        st.metric(
            label="Annual Revenue at Risk",
            value=f"${churn_arr/1e6:.2f}M",
            delta=f"${churn_mrr:,.0f} monthly",
            delta_color="inverse"
        )
    with c4:
        delta_col = "normal" if ltv_cac >= 3 else "inverse"
        st.metric(
            label="LTV:CAC Ratio",
            value=f"{ltv_cac}:1",
            delta="Benchmark: 3.0:1 minimum",
            delta_color=delta_col
        )

    st.markdown("---")

    # ── BUSINESS HEALTH SCORE ─────────────────────────────
    st.subheader("Business Health Score")

    churn_score     = max(0, 100 - (churn_rate * 2.5))
    retention_score = rev_retention
    ltv_score       = min(100, ltv_cac * 20)
    health_score    = round(
        churn_score * 0.4 +
        retention_score * 0.35 +
        ltv_score * 0.25, 1
    )

    health_colour = (
        '#e74c3c' if health_score < 50
        else '#e67e22' if health_score < 70
        else '#27ae60'
    )
    health_status = (
        'Critical' if health_score < 50
        else 'Needs Improvement' if health_score < 70
        else 'Healthy'
    )

    h_col1, h_col2 = st.columns([1, 2])

    with h_col1:
        fig_health = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': health_colour},
                'steps': [
                    {'range': [0,  50],
                     'color': 'rgba(231,76,60,0.15)'},
                    {'range': [50, 70],
                     'color': 'rgba(230,126,34,0.15)'},
                    {'range': [70, 100],
                     'color': 'rgba(39,174,96,0.15)'},
                ]
            },
            title={'text': f"Status: {health_status}"}
        ))
        fig_health.update_layout(
            height=220,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        st.plotly_chart(
            fig_health,
            use_container_width=True
        )

    with h_col2:
        st.markdown("**Health score breakdown:**")
        score_df = pd.DataFrame({
            'Metric': [
                'Churn Rate Score (40%)',
                'Revenue Retention (35%)',
                'LTV:CAC Score (25%)'
            ],
            'Score': [
                round(churn_score, 1),
                round(retention_score, 1),
                round(ltv_score, 1)
            ],
            'Status': [
                'Critical' if churn_score < 50
                else 'Needs Work' if churn_score < 70
                else 'Good',
                'Critical' if retention_score < 70
                else 'Needs Work' if retention_score < 85
                else 'Good',
                'Marginal' if ltv_score < 60
                else 'Good'
            ]
        })
        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True
        )
        st.markdown(
            f"Overall health: **{health_score}/100** — "
            f"**{health_status}**  \n"
            f"Improving churn from {churn_rate}% to 10% "
            f"would raise health score by approximately "
            f"{round((churn_rate - 10) * 2.5 * 0.4, 1)} points."
        )

    st.markdown("---")

    # ── AI BRIEFING ───────────────────────────────────────
    st.subheader("AI-Generated Executive Briefing")

    if briefing:
        # ── CLEAN FILE HEADER LINES ───────────────────────
        # Remove the file header we added when saving
        # Keep only the actual AI-generated content
        clean_lines = []
        for line in briefing.split('\n'):
            stripped = line.strip()
            if (stripped.startswith('AI SaaS') or
                    stripped.startswith('Executive Briefing —') or
                    stripped.startswith('===') or
                    stripped == 'AI SaaS GROWTH INTELLIGENCE SYSTEM'):
                continue
            clean_lines.append(line)
        clean_briefing = '\n'.join(clean_lines).strip()

        # ── PARSE INTO SECTIONS ───────────────────────────
        # Split the briefing into named sections
        # so we can render each one with proper formatting
        section_headers = [
            'SITUATION SUMMARY',
            'KEY FINDINGS',
            'IMMEDIATE RISK',
            'RECOMMENDED INTERVENTIONS',
            'PROJECTED OUTCOME'
        ]

        # Split by section headers to get content blocks
        sections = {}
        current_section = 'intro'
        current_lines   = []

        for line in clean_briefing.split('\n'):
            stripped = line.strip()
            # Remove ** markdown bold markers from headers
            clean_stripped = stripped.replace('**', '').strip()

            if clean_stripped in section_headers:
                # Save previous section
                sections[current_section] = '\n'.join(
                    current_lines
                ).strip()
                current_section = clean_stripped
                current_lines   = []
            else:
                current_lines.append(line)

        # Save last section
        sections[current_section] = '\n'.join(
            current_lines
        ).strip()

        # ── RENDER STYLED BRIEFING ────────────────────────
        

        st.markdown(
            "<hr style='margin:0 0 16px 0;'>",
            unsafe_allow_html=True
        )

        # Render each section
        section_icons = {
            'SITUATION SUMMARY'        : '📊',
            'KEY FINDINGS'             : '🔍',
            'IMMEDIATE RISK'           : '⚠️',
            'RECOMMENDED INTERVENTIONS': '✅',
            'PROJECTED OUTCOME'        : '📈'
        }

        for header in section_headers:
            content = sections.get(header, '').strip()
            if not content:
                continue

            icon = section_icons.get(header, '•')

            st.markdown(f"**{icon} {header}**")

            # ── SPECIAL HANDLING FOR INTERVENTIONS ───────
            # Reformat the verbose bullet structure into
            # compact single-line format:
            # "Segment — Action — Projected impact"
            if header == 'RECOMMENDED INTERVENTIONS':
                lines = [
                    l.strip() for l in content.split('\n')
                    if l.strip()
                ]

                compact_bullets = []
                for line in lines:
                    # Remove leading * or - bullet markers
                    line = line.lstrip('*- ').strip()
                    # Remove ** bold markers
                    line = line.replace('**', '').strip()

                    if not line:
                        continue

                    # Extract Segment, Action, Impact
                    # from the verbose Gemini format
                    segment = ''
                    action  = ''
                    impact  = ''

                    if 'Segment:' in line:
                        parts = line.split('Action:')
                        seg_raw = parts[0].replace(
                            'Segment:', ''
                        ).strip().rstrip('.')
                        segment = seg_raw

                        if len(parts) > 1:
                            action_parts = parts[1].split(
                                'Projected'
                            )
                            action = action_parts[0].strip(
                            ).rstrip('.')

                            if len(action_parts) > 1:
                                impact_raw = action_parts[
                                    1
                                ].replace(
                                    'dollar impact:', ''
                                ).replace(
                                    'impact:', ''
                                ).strip()
                                # Extract just the dollar amount
                                import re
                                dollar_match = re.search(
                                    r'\$[\d,]+', impact_raw
                                )
                                if dollar_match:
                                    impact = (
                                        f"saves "
                                        f"{dollar_match.group()}"
                                        f"/year"
                                    )
                                else:
                                    impact = impact_raw[:60]

                        if segment and action:
                            compact_bullets.append(
                                f"**{segment}** — "
                                f"{action}. "
                                f"*{impact}*"
                            )
                        else:
                            compact_bullets.append(
                                f"• {line}"
                            )
                    else:
                        # Line does not follow verbose format
                        # render as-is
                        compact_bullets.append(f"• {line}")

                for bullet in compact_bullets:
                    st.markdown(f"- {bullet}")

            else:
                # ── ALL OTHER SECTIONS ────────────────────
                # Clean ** markers and render as markdown
                clean_lines_out = []
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        clean_lines_out.append('')
                        continue
                    # Remove stray ** around section
                    # header text that leaked into content
                    if line.replace('**','').strip() in \
                       section_headers:
                        continue
                    clean_lines_out.append(line)

                clean_content = '\n'.join(
                    clean_lines_out
                ).strip()
                st.markdown(clean_content)

            st.markdown("")

        

    else:
        st.warning(
            "Briefing not found. "
            "Run Phase 12 notebook to generate."
        )

    st.markdown("---")

    # ── RISK OVERVIEW ─────────────────────────────────────
    st.subheader("Risk Distribution Overview")

    col_left, col_right = st.columns(2)
    levels  = ['Critical', 'High', 'Medium', 'Low']
    counts  = [risk_counts[l] for l in levels]
    colours = ['#e74c3c', '#e67e22', '#f1c40f', '#27ae60']

    with col_left:
        fig_donut = go.Figure(data=[go.Pie(
            labels=levels,
            values=counts,
            hole=0.45,
            marker=dict(colors=colours),
            textinfo='label+percent',
            textfont_size=13
        )])
        fig_donut.update_layout(
            title="Customers by Risk Level",
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
            height=320
        )
        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )

    with col_right:
        churn_by_level = []
        for level in levels:
            sub  = risk_df[risk_df['risk_level'] == level]
            rate = round(
                sub['actual_churn'].mean() * 100, 1
            )
            churn_by_level.append(rate)

        fig_bar = go.Figure(data=[go.Bar(
            x=levels,
            y=churn_by_level,
            marker_color=colours,
            text=[f"{v}%" for v in churn_by_level],
            textposition='outside'
        )])
        fig_bar.add_hline(
            y=26.58,
            line_dash="dash",
            line_color="gray",
            annotation_text="Overall avg 26.58%"
        )
        fig_bar.update_layout(
            title="Actual Churn Rate by Risk Level",
            yaxis_title="Churn Rate (%)",
            showlegend=False,
            margin=dict(t=40, b=10),
            height=320
        )
        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

    st.markdown("---")

    # ── TOP 10 PRIORITY ───────────────────────────────────
    st.subheader("Top 10 Customers to Contact This Week")

    top10 = risk_df.nlargest(10, 'risk_score')[[
        'priority_rank', 'customerID', 'risk_score',
        'risk_level', 'monthly_revenue',
        'tenure_months', 'contract_type'
    ]].reset_index(drop=True)

    top10['monthly_revenue'] = top10[
        'monthly_revenue'
    ].apply(lambda x: f"${x:.2f}")

    top10.columns = [
        'Rank', 'Customer ID', 'Risk Score',
        'Risk Level', 'Monthly Value',
        'Tenure (months)', 'Contract'
    ]
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )

    # Download full report
    csv_full = risk_df[[
        'customerID', 'risk_score', 'risk_level',
        'churn_drivers', 'recommended_action',
        'monthly_revenue', 'annual_revenue_risk',
        'tenure_months', 'contract_type'
    ]].to_csv(index=False)

    st.download_button(
        label="Download complete risk report (CSV)",
        data=csv_full,
        file_name="saas_risk_report.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER RISK EXPLORER
# ══════════════════════════════════════════════════════════
elif page == "Customer Risk Explorer":

    st.title("Customer Risk Explorer")
    st.markdown(
        "Search any customer to see their complete "
        "risk profile, churn drivers, and recommended action."
    )
    st.markdown("---")

    col_search, col_select = st.columns(2)

    with col_search:
        search_id = st.text_input(
            "Enter Customer ID",
            placeholder="e.g. 9300-AGZNL"
        ).strip().upper()

    with col_select:
        top_ids = risk_df.nlargest(
            20, 'risk_score'
        )['customerID'].tolist()
        selected = st.selectbox(
            "Or select from top 20 critical customers",
            ["— select —"] + top_ids
        )
        if selected != "— select —":
            search_id = selected

    if search_id:
        row_match = risk_df[
            risk_df['customerID'].str.upper() == search_id
        ]

        if len(row_match) == 0:
            st.error(
                f"Customer '{search_id}' not found."
            )
        else:
            row = row_match.iloc[0]

            level_icons = {
                'Critical': '🔴',
                'High'    : '🟠',
                'Medium'  : '🟡',
                'Low'     : '🟢'
            }
            icon = level_icons.get(row['risk_level'], '⚪')

            st.markdown("---")

            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Risk Score",
                f"{row['risk_score']}/100"
            )
            m2.metric(
                "Risk Level",
                f"{icon} {row['risk_level']}"
            )
            m3.metric(
                "Churn Probability",
                f"{round(row['ensemble_probability']*100,1)}%"
            )
            m4.metric(
                "Priority Rank",
                f"#{int(row['priority_rank'])} "
                f"of {len(risk_df):,}"
            )

            st.markdown("---")

            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("Customer Profile")
                actual_txt = (
                    "🔴 Churned"
                    if row['actual_churn'] == 1
                    else "🟢 Currently Active"
                )
                st.markdown(
                    f"**Customer ID:** "
                    f"{row['customerID']}  \n"
                    f"**Tenure:** "
                    f"{int(row['tenure_months'])} months  \n"
                    f"**Contract:** "
                    f"{row['contract_type']}  \n"
                    f"**Payment:** "
                    f"{row['payment_method']}  \n"
                    f"**Monthly Value:** "
                    f"${row['monthly_revenue']:.2f}  \n"
                    f"**Annual Value:** "
                    f"${row['annual_revenue_risk']:,.2f}  \n"
                    f"**Actual Outcome:** {actual_txt}"
                )

            with col_r:
                st.subheader("Why at Risk")
                drivers = row['churn_drivers'].split(' | ')
                for i, d in enumerate(drivers, 1):
                    st.markdown(f"**{i}.** {d}")

                st.markdown("---")
                st.subheader("Recommended Action")
                for action in row[
                    'recommended_action'
                ].split('. '):
                    if action.strip():
                        st.markdown(
                            f"• {action.strip()}"
                        )

            # Gauge
            st.markdown("---")
            g_col1, g_col2 = st.columns([1, 1])

            with g_col1:
                st.subheader("Risk Score Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=float(row['risk_score']),
                    delta={
                        'reference'  : 30,
                        'increasing' : {'color': '#e74c3c'}
                    },
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar' : {'color': '#2c3e50'},
                        'steps': [
                            {'range': [0,  30],
                             'color': 'rgba(39,174,96,0.2)'},
                            {'range': [30, 50],
                             'color': 'rgba(241,196,15,0.2)'},
                            {'range': [50, 70],
                             'color': 'rgba(230,126,34,0.2)'},
                            {'range': [70, 100],
                             'color': 'rgba(231,76,60,0.2)'},
                        ],
                        'threshold': {
                            'line': {
                                'color': 'red',
                                'width': 4
                            },
                            'thickness': 0.75,
                            'value': float(row['risk_score'])
                        }
                    },
                    title={'text': "Churn Risk Score"}
                ))
                fig_gauge.update_layout(height=280)
                st.plotly_chart(
                    fig_gauge,
                    use_container_width=True
                )

            # What-if simulator
            with g_col2:
                st.subheader("What-If Simulator")
                st.markdown(
                    "See how risk score changes "
                    "with different contract or payment."
                )

                contract_options = [
                    'Month-to-month',
                    'One year',
                    'Two year'
                ]
                current_contract = row['contract_type']
                default_idx = (
                    contract_options.index(current_contract)
                    if current_contract in contract_options
                    else 0
                )

                new_contract = st.selectbox(
                    "Change contract to:",
                    contract_options,
                    index=default_idx
                )

                payment_options = [
                    'Electronic check',
                    'Mailed check',
                    'Bank transfer (automatic)',
                    'Credit card (automatic)'
                ]
                new_payment = st.selectbox(
                    "Change payment to:",
                    payment_options
                )

                score_delta = 0
                reasons = []

                if (new_contract == 'Two year' and
                        current_contract == 'Month-to-month'):
                    score_delta -= 35
                    reasons.append(
                        "Two-year contract: -35 pts"
                    )
                elif (new_contract == 'One year' and
                      current_contract == 'Month-to-month'):
                    score_delta -= 20
                    reasons.append(
                        "One-year contract: -20 pts"
                    )

                if ('automatic' in new_payment.lower() and
                        'automatic' not in str(
                            row['payment_method']
                        ).lower()):
                    score_delta -= 15
                    reasons.append(
                        "Auto payment: -15 pts"
                    )

                new_score = max(
                    0, min(
                        100,
                        float(row['risk_score']) + score_delta
                    )
                )

                if score_delta < 0:
                    st.success(
                        f"Risk score: "
                        f"**{row['risk_score']}** → "
                        f"**{round(new_score, 1)}**  \n"
                        f"Improvement: "
                        f"**{abs(score_delta)} points**  \n"
                        + "  \n".join(reasons)
                    )
                elif score_delta == 0:
                    st.info(
                        f"No change from current score "
                        f"of {row['risk_score']}.  \n"
                        f"Try upgrading contract type."
                    )


# ══════════════════════════════════════════════════════════
# PAGE 3 — SEGMENT ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "Segment Analysis":

    st.title("Segment Analysis")
    st.markdown(
        "Explore churn patterns across customer segments. "
        "Filters update all charts in real time."
    )
    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns(3)

    with f1:
        contract_filter = st.multiselect(
            "Contract Type",
            options=df['Contract'].unique().tolist(),
            default=df['Contract'].unique().tolist()
        )
    with f2:
        payment_filter = st.multiselect(
            "Payment Method",
            options=df['PaymentMethod'].unique().tolist(),
            default=df['PaymentMethod'].unique().tolist()
        )
    with f3:
        tenure_range = st.slider(
            "Tenure Range (months)",
            min_value=0, max_value=72,
            value=(0, 72)
        )

    mask = (
        df['Contract'].isin(contract_filter) &
        df['PaymentMethod'].isin(payment_filter) &
        df['tenure'].between(
            tenure_range[0], tenure_range[1]
        )
    )
    df_f = df[mask].copy()

    if len(df_f) == 0:
        st.warning("No customers match the selected filters.")
        st.stop()

    f_churn   = round(df_f['Churn'].mean() * 100, 2)
    f_count   = len(df_f)
    f_churned = int(df_f['Churn'].sum())
    f_rev     = round(
        df_f[df_f['Churn']==1
             ]['MonthlyCharges'].sum() * 12, 0
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{f_count:,}")
    m2.metric("Churn Rate", f"{f_churn}%",
              delta=f"{round(f_churn-churn_rate,2)}% vs avg",
              delta_color="inverse")
    m3.metric("Churned", f"{f_churned:,}")
    m4.metric("Annual Rev at Risk", f"${f_rev:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract Type")
        ct = df_f.groupby('Contract')[
            'Churn'
        ].mean().mul(100).round(1).reset_index()
        ct.columns = ['Contract', 'Churn Rate (%)']
        fig1 = px.bar(
            ct, x='Contract', y='Churn Rate (%)',
            color='Churn Rate (%)',
            color_continuous_scale=[
                '#27ae60', '#f1c40f', '#e74c3c'
            ],
            text='Churn Rate (%)'
        )
        fig1.update_traces(
            texttemplate='%{text}%',
            textposition='outside'
        )
        fig1.update_layout(
            showlegend=False, height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Churn by Payment Method")
        pm = df_f.groupby('PaymentMethod')[
            'Churn'
        ].mean().mul(100).round(1).reset_index()
        pm.columns = ['Payment Method', 'Churn Rate (%)']
        pm = pm.sort_values('Churn Rate (%)', ascending=True)
        fig2 = px.bar(
            pm, x='Churn Rate (%)', y='Payment Method',
            orientation='h',
            color='Churn Rate (%)',
            color_continuous_scale=[
                '#27ae60', '#f1c40f', '#e74c3c'
            ],
            text='Churn Rate (%)'
        )
        fig2.update_traces(
            texttemplate='%{text}%',
            textposition='outside'
        )
        fig2.update_layout(
            showlegend=False, height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tenure lifecycle
    st.subheader("Churn Rate Across Tenure Groups")

    def tgrp(t):
        if t <= 12:   return '0-12m'
        elif t <= 24: return '13-24m'
        elif t <= 36: return '25-36m'
        elif t <= 48: return '37-48m'
        elif t <= 60: return '49-60m'
        else:         return '61-72m'

    df_f['tgrp'] = df_f['tenure'].apply(tgrp)
    order = [
        '0-12m','13-24m','25-36m',
        '37-48m','49-60m','61-72m'
    ]
    tg = df_f.groupby('tgrp', observed=True)[
        'Churn'
    ].mean().mul(100).round(1).reset_index()
    tg.columns = ['Tenure Group', 'Churn Rate (%)']
    tg['Tenure Group'] = pd.Categorical(
        tg['Tenure Group'],
        categories=order, ordered=True
    )
    tg = tg.sort_values('Tenure Group')

    fig3 = px.line(
        tg, x='Tenure Group', y='Churn Rate (%)',
        markers=True,
        color_discrete_sequence=['#e74c3c'],
        text='Churn Rate (%)'
    )
    fig3.update_traces(
        textposition='top center',
        texttemplate='%{text}%'
    )
    fig3.add_hline(
        y=churn_rate, line_dash="dash",
        line_color="gray",
        annotation_text=f"Overall avg {churn_rate}%"
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, use_container_width=True)

    # Engagement vs churn
    st.subheader("Product Stickiness — Engagement vs Churn")

    service_cols = [
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    df_f = df_f.copy()
    df_f['eng'] = df_f[service_cols].apply(
        lambda col: (col == 'Yes').astype(int)
    ).sum(axis=1)

    eng = df_f.groupby('eng')[
        'Churn'
    ].mean().mul(100).round(1).reset_index()
    eng.columns = ['Services Used', 'Churn Rate (%)']

    fig4 = px.bar(
        eng, x='Services Used', y='Churn Rate (%)',
        color='Churn Rate (%)',
        color_continuous_scale=[
            '#27ae60', '#f1c40f', '#e74c3c'
        ],
        text='Churn Rate (%)'
    )
    fig4.update_traces(
        texttemplate='%{text}%',
        textposition='outside'
    )
    fig4.update_layout(
        showlegend=False, height=320,
        xaxis_title="Number of Services Used (0-7)",
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — REVENUE CALCULATOR
# ══════════════════════════════════════════════════════════
elif page == "Revenue Calculator":

    st.title("Revenue Impact Calculator")
    st.markdown(
        "Calculate revenue saved at different churn "
        "reduction targets. Use this to build the "
        "business case for retention investment."
    )
    st.markdown("---")

    st.subheader("Set Your Churn Reduction Target")

    reduction_pct = st.slider(
        "Target churn reduction (%)",
        min_value=5, max_value=75,
        value=20, step=5,
        help="What % of current churners do you want to retain?"
    )

    customers_saved = round(
        total_churned * reduction_pct / 100
    )
    revenue_saved   = round(
        customers_saved * avg_monthly * 12, 0
    )
    new_churn_rate  = round(
        churn_rate * (1 - reduction_pct / 100), 2
    )
    new_mrr         = round(
        mrr + customers_saved * avg_monthly, 0
    )

    st.markdown("---")
    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Customers Retained",
        f"{customers_saved:,}",
        f"of {total_churned:,} churners"
    )
    r2.metric(
        "Annual Revenue Saved",
        f"${revenue_saved:,.0f}",
        f"${round(revenue_saved/12):,}/month",
        delta_color="normal"
    )
    r3.metric(
        "New Churn Rate",
        f"{new_churn_rate}%",
        f"{round(churn_rate-new_churn_rate,2)}% improvement",
        delta_color="normal"
    )
    r4.metric(
        "New MRR",
        f"${new_mrr:,.0f}",
        f"+${round(customers_saved*avg_monthly):,}/month",
        delta_color="normal"
    )

    st.markdown("---")
    st.subheader("All Scenarios — Revenue Saved")

    rows = []
    for pct in range(5, 76, 5):
        saved  = round(total_churned * pct / 100)
        rev    = round(saved * avg_monthly * 12, 0)
        new_cr = round(churn_rate * (1 - pct / 100), 2)
        rows.append({
            'Reduction %'   : pct,
            'Customers Saved': saved,
            'Revenue Saved' : rev,
            'New Churn Rate': new_cr
        })

    scen_df = pd.DataFrame(rows)

    fig_s = px.bar(
        scen_df,
        x='Reduction %', y='Revenue Saved',
        color='Revenue Saved',
        color_continuous_scale=['#f1c40f', '#27ae60'],
        text='Revenue Saved'
    )
    fig_s.update_traces(
        texttemplate='$%{text:,.0f}',
        textposition='outside',
        textfont_size=9
    )
    fig_s.add_vline(
        x=reduction_pct,
        line_dash="dash", line_color="red",
        annotation_text=f"Your target: {reduction_pct}%"
    )
    fig_s.update_layout(
        showlegend=False, height=420,
        yaxis_title="Annual Revenue Saved ($)"
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # ROI calculator
    st.markdown("---")
    st.subheader("Return on Investment Calculator")

    programme_cost = st.number_input(
        "Estimated retention programme cost ($/year)",
        min_value=0, max_value=500000,
        value=50000, step=5000
    )

    if programme_cost > 0:
        net_gain       = revenue_saved - programme_cost
        roi            = round(net_gain / programme_cost * 100, 1)
        payback_months = round(
            programme_cost / (revenue_saved / 12), 1
        ) if revenue_saved > 0 else 0

        roi_c1, roi_c2, roi_c3 = st.columns(3)
        roi_c1.metric(
            "Net Revenue Gain",
            f"${net_gain:,.0f}",
            f"After ${programme_cost:,} cost",
            delta_color="normal" if net_gain > 0
            else "inverse"
        )
        roi_c2.metric(
            "ROI",
            f"{roi}%",
            delta_color="normal" if roi > 0
            else "inverse"
        )
        roi_c3.metric(
            "Payback Period",
            f"{payback_months} months",
            delta_color="normal"
        )

    # Scenario table
    st.markdown("---")
    st.subheader("Full Scenario Table")
    st.dataframe(
        scen_df,
        use_container_width=True,
        hide_index=True
    )


# ══════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════
elif page == "Recommendation Engine":

    st.title("Recommendation Engine")
    st.markdown(
        "AI-powered retention recommendations by segment. "
        "Filter and download your action list."
    )
    st.markdown("---")

    # Segment strategy cards
    st.subheader("Segment Strategy")

    level_icons = {
        'Critical': '🔴',
        'High'    : '🟠',
        'Medium'  : '🟡',
        'Low'     : '🟢'
    }

    if recs:
        for level in ['Critical', 'High', 'Medium', 'Low']:
            icon    = level_icons.get(level, '')
            count   = risk_counts.get(level, 0)
            sub     = risk_df[risk_df['risk_level']==level]
            churn_r = round(
                sub['actual_churn'].mean() * 100, 1
            )
            rev_r   = round(
                sub['annual_revenue_risk'].sum(), 0
            )
            with st.expander(
                f"{icon} {level} Risk — "
                f"{count:,} customers | "
                f"{churn_r}% churn | "
                f"${rev_r:,.0f} at risk",
                expanded=(level == 'Critical')
            ):
                st.markdown(recs.get(level, ''))

                # Action timeline per level
                timelines = {
                    'Critical': (
                        "**Action timeline:** Personal call "
                        "within **24 hours**"
                    ),
                    'High'    : (
                        "**Action timeline:** Email + "
                        "follow-up call within **3 days**"
                    ),
                    'Medium'  : (
                        "**Action timeline:** Automated "
                        "email sequence within **1 week**"
                    ),
                    'Low'     : (
                        "**Action timeline:** Monthly "
                        "newsletter + quarterly review"
                    )
                }
                st.markdown(timelines.get(level, ''))
    else:
        st.warning(
            "Recommendations not found. "
            "Run Phase 12 notebook."
        )

    st.markdown("---")

    # Customer action list
    st.subheader("Customer Action List")

    fa_col1, fa_col2 = st.columns(2)
    with fa_col1:
        selected_levels = st.multiselect(
            "Filter by risk level",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High']
        )
    with fa_col2:
        max_rows = st.slider(
            "Customers to show",
            min_value=10, max_value=300,
            value=50
        )

    if selected_levels:
        filt = risk_df[
            risk_df['risk_level'].isin(selected_levels)
        ].nlargest(max_rows, 'risk_score')[[
            'priority_rank', 'customerID',
            'risk_score', 'risk_level',
            'monthly_revenue', 'tenure_months',
            'contract_type', 'churn_drivers',
            'recommended_action'
        ]].reset_index(drop=True)

        filt['monthly_revenue'] = filt[
            'monthly_revenue'
        ].apply(lambda x: f"${x:.2f}")

        filt.columns = [
            'Rank', 'Customer ID', 'Score', 'Level',
            'Monthly $', 'Tenure', 'Contract',
            'Why at risk', 'Action'
        ]

        # Summary before table
        total_rev_at_risk = risk_df[
            risk_df['risk_level'].isin(selected_levels)
        ]['annual_revenue_risk'].sum()

        s1, s2, s3 = st.columns(3)
        s1.metric(
            "Customers shown",
            f"{len(filt):,}"
        )
        s2.metric(
            "Total annual rev at risk",
            f"${total_rev_at_risk:,.0f}"
        )
        s3.metric(
            "Avg risk score",
            f"{filt['Score'].mean():.1f}/100"
        )

        st.dataframe(
            filt,
            use_container_width=True,
            hide_index=True,
            height=500
        )

        csv_out = filt.to_csv(index=False)
        st.download_button(
            label="Download action list as CSV",
            data=csv_out,
            file_name=f"retention_action_list_"
                      f"{'_'.join(selected_levels)}.csv",
            mime="text/csv"
        )
    else:
        st.info("Select at least one risk level above.")