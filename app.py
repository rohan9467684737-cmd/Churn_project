"""
Customer Churn Prediction — Industry-Level Streamlit App
Author: Resume Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import io
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background-color: #0f1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252836);
    border: 1px solid #2d3147;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card .value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #48c6ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #8b92a5;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-card .delta {
    font-size: 0.8rem;
    margin-top: 6px;
}

/* Risk badge */
.risk-high   { background:#ff4b4b22; border:1px solid #ff4b4b; color:#ff4b4b; border-radius:8px; padding:4px 12px; font-weight:600; }
.risk-medium { background:#ffa50022; border:1px solid #ffa500; color:#ffa500; border-radius:8px; padding:4px 12px; font-weight:600; }
.risk-low    { background:#00c85322; border:1px solid #00c853; color:#00c853; border-radius:8px; padding:4px 12px; font-weight:600; }

/* Section header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e0e4f0;
    border-left: 4px solid #6c63ff;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Gauge container */
.gauge-wrapper { text-align: center; padding: 10px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #151721; border-right: 1px solid #2d3147; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #48c6ef);
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 600; width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* DataFrame */
.dataframe { border-radius: 8px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-weight: 600; }

/* Top banner */
.top-banner {
    background: linear-gradient(135deg, #1a1c2e 0%, #252838 100%);
    border: 1px solid #2d3147;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("models/churn_pipeline.pkl")

@st.cache_data(show_spinner=False)
def load_metadata():
    return json.load(open("models/metadata.json"))

@st.cache_data(show_spinner=False)
def load_train_data():
    df = pd.read_csv("data/train.csv").dropna().sample(20000, random_state=42)
    return df

@st.cache_data(show_spinner=False)
def load_test_data():
    return pd.read_csv("data/test.csv").dropna()

try:
    model    = load_model()
    meta     = load_metadata()
    train_df = load_train_data()
    test_df  = load_test_data()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    st.error(f"❌ Could not load model artifacts: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────
#  HELPER: FEATURE ENGINEERING (must match training)
# ─────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["CustomerID"], errors="ignore", inplace=True)
    df["SpendPerTenure"]   = df["Total Spend"]   / (df["Tenure"] + 1)
    df["CallsPerTenure"]   = df["Support Calls"] / (df["Tenure"] + 1)
    df["DelayRatio"]       = df["Payment Delay"] / (df["Tenure"] + 1)
    df["EngagementScore"]  = df["Usage Frequency"] * df["Tenure"]
    df["SupportIntensity"] = df["Support Calls"]   * df["Payment Delay"]
    df["HighRiskFlag"]     = ((df["Support Calls"] > 5) & (df["Payment Delay"] > 15)).astype(int)
    df["AgeGroup"]    = pd.cut(df["Age"],    bins=[0,25,35,50,65,100],
                                labels=["lt25","25to35","35to50","50to65","gt65"]).astype(str)
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=[0,12,24,36,60,200],
                                labels=["lt1yr","1to2yr","2to3yr","3to5yr","gt5yr"]).astype(str)
    return df


def predict_single(row_dict: dict) -> tuple[float, str, str]:
    """Return (probability, risk_level, risk_label)."""
    df = pd.DataFrame([row_dict])
    df = feature_engineering(df)
    feat_cols = meta["feature_names"]
    df = df.reindex(columns=feat_cols, fill_value=0)
    prob = float(model.predict_proba(df)[0][1])
    thresh = meta["best_threshold"]
    if prob >= 0.70:
        level, label = "High",   "🔴 HIGH RISK"
    elif prob >= thresh:
        level, label = "Medium", "🟡 MEDIUM RISK"
    else:
        level, label = "Low",    "🟢 LOW RISK"
    return prob, level, label


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = feature_engineering(df.copy())
    feat_cols = meta["feature_names"]
    df_fe = df_fe.reindex(columns=feat_cols, fill_value=0)
    probs  = model.predict_proba(df_fe)[:, 1]
    thresh = meta["best_threshold"]
    preds  = (probs >= thresh).astype(int)
    result = df.copy()
    result["Churn_Probability"] = probs.round(4)
    result["Churn_Prediction"]  = preds
    result["Risk_Level"]        = np.where(probs >= 0.70, "High",
                                  np.where(probs >= thresh, "Medium", "Low"))
    return result


# ─────────────────────────────────────────────────────────────
#  COLOR PALETTE
# ─────────────────────────────────────────────────────────────
PURPLE  = "#6c63ff"
CYAN    = "#48c6ef"
RED     = "#ff4b4b"
GREEN   = "#00c853"
ORANGE  = "#ffa500"
BG      = "#0f1117"
CARD_BG = "#1e2130"
TEXT    = "#e0e4f0"
MUTED   = "#8b92a5"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#2d3147", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "grid.color": "#2d3147",
    "grid.alpha": 0.5, "font.family": "sans-serif",
})


# ─────────────────────────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.4rem;'>📉</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e0e4f0; margin-top:4px;'>ChurnIQ</div>
        <div style='font-size:0.75rem; color:#6c63ff; letter-spacing:0.12em; margin-top:2px;'>INTELLIGENCE PLATFORM</div>
    </div>
    <hr style='border-color:#2d3147; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "📊  EDA Dashboard", "🔮  Live Prediction", "📁  Batch Prediction", "📈  Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2d3147;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#8b92a5; line-height:1.7;'>
        <b style='color:#e0e4f0;'>Model</b><br>{meta['model_name']}<br><br>
        <b style='color:#e0e4f0;'>ROC-AUC</b><br>{meta['test_roc_auc']:.4f}<br><br>
        <b style='color:#e0e4f0;'>Train Rows</b><br>{meta['total_train_rows']:,}<br><br>
        <b style='color:#e0e4f0;'>Test Rows</b><br>{meta['test_size']:,}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div class='top-banner'>
        <div style='font-size:2rem; font-weight:800; color:#e0e4f0;'>Customer Churn Intelligence Platform</div>
        <div style='font-size:1rem; color:#8b92a5; margin-top:6px;'>
            End-to-end ML system for predicting, explaining, and acting on customer churn risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("ROC-AUC",   f"{meta['test_roc_auc']:.4f}",   "Model discrimination"),
        ("F1 Score",  f"{meta['test_f1']:.4f}",         "Harmonic mean P/R"),
        ("Precision", f"{meta['test_precision']:.4f}",  "Positive predictive value"),
        ("Recall",    f"{meta['test_recall']:.4f}",     "Sensitivity / TPR"),
        ("PR-AUC",    f"{meta['test_pr_auc']:.4f}",     "Precision-Recall area"),
    ]
    for col, (label, value, desc) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='value'>{value}</div>
                <div class='label'>{label}</div>
                <div class='delta' style='color:#8b92a5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Dataset stats + Confusion matrix side by side
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
        eda = meta["eda_stats"]
        churned   = eda["churn_dist"].get("1", eda["churn_dist"].get(1, 0))
        retained  = eda["churn_dist"].get("0", eda["churn_dist"].get(0, 0))
        total     = churned + retained

        stats_data = {
            "Metric": ["Total Train Rows", "Test Rows", "Features Engineered",
                       "Churned (Train)", "Retained (Train)", "Train Churn Rate", "Test Churn Rate"],
            "Value":  [f"{meta['total_train_rows']:,}", f"{meta['test_size']:,}",
                       str(len(meta['feature_names'])),
                       f"{churned:,}", f"{retained:,}",
                       f"{meta['churn_rate_train']:.1%}", f"{meta['churn_rate_test']:.1%}"],
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # Class distribution donut
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor(CARD_BG)
        ax.set_facecolor(CARD_BG)
        sizes  = [churned, retained]
        colors = [RED, GREEN]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=["Churned", "Retained"], colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=CARD_BG, linewidth=2),
        )
        for t in texts:     t.set_color(TEXT)
        for t in autotexts: t.set_color("white"); t.set_fontweight("bold")
        ax.set_title("Train Class Distribution", color=TEXT, fontweight="bold", pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = np.array(meta["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(CARD_BG)
        ax.set_facecolor(CARD_BG)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Predicted\nRetained", "Predicted\nChurned"],
                    yticklabels=["Actual\nRetained", "Actual\nChurned"],
                    linewidths=2, linecolor=CARD_BG,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title("Test Set Confusion Matrix", color=TEXT, fontweight="bold", pad=12)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.set_xlabel("", color=TEXT); ax.set_ylabel("", color=TEXT)
        total_test = cm.sum()
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        ax.text(0.5, -0.12,
            f"TP={tp:,}   FP={fp:,}   FN={fn:,}   TN={tn:,}",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Threshold info
        st.info(f"**Optimal Decision Threshold:** `{meta['best_threshold']:.3f}`  \n"
                f"Tuned on validation F1. Adjust in the Live Prediction page.")

    # Feature importance bar chart
    st.markdown("<div class='section-header'>Top Feature Importances</div>", unsafe_allow_html=True)
    fi = pd.DataFrame(meta["feature_importances"]).head(15)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    colors_bar = [PURPLE if i < 3 else CYAN if i < 8 else MUTED for i in range(len(fi))]
    bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors_bar[::-1], height=0.65)
    ax.set_xlabel("Importance (|coefficient|)", color=MUTED, fontsize=9)
    ax.set_title("Feature Importance — Top 15", color=TEXT, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ════════════════════════════════════════════════════════════
#  PAGE: EDA DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "📊  EDA Dashboard":
    st.markdown("## 📊 Exploratory Data Analysis")
    st.caption(f"Visualizing patterns in {meta['total_train_rows']:,} training records")

    eda = meta["eda_stats"]
    tab1, tab2, tab3 = st.tabs(["📦 Distributions", "🔗 Churn Drivers", "📐 Segment Analysis"])

    # ── TAB 1: Distributions ──────────────────────────────
    with tab1:
        numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                        "Payment Delay", "Total Spend", "Last Interaction"]
        n_cols = 4
        n_rows = -(-len(numeric_cols) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
        fig.patch.set_facecolor(BG)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            ax.set_facecolor(CARD_BG)
            churned_vals  = train_df[train_df["Churn"] == 1][col].dropna()
            retained_vals = train_df[train_df["Churn"] == 0][col].dropna()
            ax.hist(retained_vals, bins=30, alpha=0.65, color=GREEN,  label="Retained", density=True)
            ax.hist(churned_vals,  bins=30, alpha=0.65, color=RED,    label="Churned",  density=True)
            ax.set_title(col, color=TEXT, fontweight="bold", fontsize=10)
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(alpha=0.2)
            if i == 0:
                ax.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8)

        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions by Churn Status", color=TEXT,
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Stats table
        st.markdown("#### 📊 Summary Statistics (Churned vs Retained)")
        churn_avg = eda["churn_avg"]
        rows = []
        for col in numeric_cols:
            rows.append({
                "Feature": col,
                "Churned (mean)": churn_avg[col]["churned"],
                "Retained (mean)": churn_avg[col]["retained"],
                "Diff": round(churn_avg[col]["churned"] - churn_avg[col]["retained"], 2),
            })
        df_stats = pd.DataFrame(rows)
        st.dataframe(df_stats.style.background_gradient(subset=["Diff"], cmap="RdYlGn_r"),
                     use_container_width=True, hide_index=True)

    # ── TAB 2: Churn Drivers ──────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Churn rate by Contract Length
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(CARD_BG)
            ax.set_facecolor(CARD_BG)
            contract_data = eda["churn_by_contract"]
            keys = list(contract_data.keys())
            vals = [contract_data[k] for k in keys]
            bar_colors = [RED if v > 0.5 else ORANGE if v > 0.35 else GREEN for v in vals]
            bars = ax.bar(keys, vals, color=bar_colors, width=0.5, edgecolor=CARD_BG)
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color=MUTED, linestyle="--", alpha=0.5, linewidth=1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{v:.1%}", ha="center", fontsize=10, color=TEXT, fontweight="bold")
            ax.set_title("Churn Rate by Contract Type", color=TEXT, fontweight="bold")
            ax.set_ylabel("Churn Rate", color=MUTED)
            ax.set_xlabel("Contract Length", color=MUTED)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            # Churn rate by Subscription Type
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor(CARD_BG)
            ax.set_facecolor(CARD_BG)
            sub_data = eda["churn_by_subscription"]
            keys = list(sub_data.keys())
            vals = [sub_data[k] for k in keys]
            bar_colors = [RED if v > 0.5 else ORANGE if v > 0.35 else GREEN for v in vals]
            bars = ax.bar(keys, vals, color=bar_colors, width=0.5, edgecolor=CARD_BG)
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color=MUTED, linestyle="--", alpha=0.5, linewidth=1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{v:.1%}", ha="center", fontsize=10, color=TEXT, fontweight="bold")
            ax.set_title("Churn Rate by Subscription Type", color=TEXT, fontweight="bold")
            ax.set_ylabel("Churn Rate", color=MUTED)
            ax.set_xlabel("Subscription Type", color=MUTED)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Support Calls vs Churn
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor(BG)
        for ax in axes: ax.set_facecolor(CARD_BG)

        # Payment Delay vs Churn binned
        for idx, (col, xlabel) in enumerate([("Support Calls","Support Calls"), ("Payment Delay","Payment Delay (days)")]):
            ax = axes[idx]
            means = train_df.groupby(col)["Churn"].mean()
            ax.bar(means.index, means.values,
                   color=[RED if v > 0.5 else ORANGE if v > 0.35 else GREEN for v in means.values],
                   alpha=0.85)
            ax.axhline(0.5, color=MUTED, linestyle="--", alpha=0.4, linewidth=1)
            ax.set_title(f"Churn Rate by {xlabel}", color=TEXT, fontweight="bold")
            ax.set_xlabel(xlabel, color=MUTED)
            ax.set_ylabel("Churn Rate", color=MUTED)
            ax.set_ylim(0, 1)
            ax.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── TAB 3: Segments ──────────────────────────────────
    with tab3:
        # Correlation heatmap
        st.markdown("#### 🔥 Feature Correlation Matrix")
        num_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                    "Payment Delay", "Total Spend", "Last Interaction", "Churn"]
        corr = train_df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, linecolor="#2d3147",
                    annot_kws={"size": 9}, vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Heatmap", color=TEXT, fontweight="bold", pad=12)
        ax.tick_params(colors=TEXT, labelsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

        # Spend vs Tenure scatter
        st.markdown("#### 💸 Total Spend vs Tenure (by Churn)")
        sample = train_df.sample(min(3000, len(train_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
        for churn_val, color, label in [(0, GREEN, "Retained"), (1, RED, "Churned")]:
            sub = sample[sample["Churn"] == churn_val]
            ax.scatter(sub["Tenure"], sub["Total Spend"],
                       c=color, alpha=0.35, s=14, label=label, edgecolors="none")
        ax.set_xlabel("Tenure (months)", color=MUTED)
        ax.set_ylabel("Total Spend ($)", color=MUTED)
        ax.set_title("Total Spend vs Tenure", color=TEXT, fontweight="bold")
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True); plt.close()


# ════════════════════════════════════════════════════════════
#  PAGE: LIVE PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":
    st.markdown("## 🔮 Live Customer Churn Prediction")
    st.caption("Enter customer details to get an instant churn probability with risk breakdown.")

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown("### Customer Profile")
        with st.container():
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                age     = st.slider("Age", 18, 80, 35)
                tenure  = st.slider("Tenure (months)", 1, 60, 24)
                usage   = st.slider("Usage Frequency", 1, 30, 14)
            with r1c2:
                support = st.slider("Support Calls", 0, 10, 3)
                delay   = st.slider("Payment Delay (days)", 0, 30, 10)
                spend   = st.number_input("Total Spend ($)", 50, 1000, 500, step=10)
            last_int = st.slider("Last Interaction (days ago)", 1, 30, 10)
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                gender   = st.selectbox("Gender",   ["Male", "Female"])
            with r2c2:
                sub_type = st.selectbox("Subscription", ["Basic", "Standard", "Premium"])
            with r2c3:
                contract = st.selectbox("Contract",  ["Monthly", "Quarterly", "Annual"])

        # Custom threshold slider
        st.markdown("---")
        user_thresh = st.slider("Decision Threshold", 0.1, 0.9,
                                 float(meta["best_threshold"]), 0.01,
                                 help="Lower = more sensitive (catch more churners, more false positives)")

        predict_btn = st.button("🔮  Predict Churn Risk", use_container_width=True)

    # Build input
    input_data = {
        "Age": age, "Gender": gender, "Tenure": tenure,
        "Usage Frequency": usage, "Support Calls": support,
        "Payment Delay": delay, "Subscription Type": sub_type,
        "Contract Length": contract, "Total Spend": spend,
        "Last Interaction": last_int,
    }

    if predict_btn or True:  # Show result by default
        prob, level, label = predict_single(input_data)
        # Apply user threshold
        if prob >= 0.70:
            level, label = "High", "🔴 HIGH RISK"
        elif prob >= user_thresh:
            level, label = "Medium", "🟡 MEDIUM RISK"
        else:
            level, label = "Low", "🟢 LOW RISK"

        with col_result:
            st.markdown("### Prediction Result")

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5.5, 3.5), subplot_kw=dict(aspect="equal"))
            fig.patch.set_facecolor(CARD_BG)
            ax.set_facecolor(CARD_BG)

            # Background arc (grey)
            theta_start, theta_end = np.pi, 0
            theta = np.linspace(theta_start, theta_end, 200)
            r_outer, r_inner = 1.0, 0.60
            ax.fill_between(np.cos(theta), np.sin(theta),
                             r_inner * np.sin(theta), alpha=0.12, color=MUTED)
            ax.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=2, alpha=0.3)
            ax.plot(r_inner*np.cos(theta), r_inner*np.sin(theta), color=MUTED, lw=2, alpha=0.3)

            # Colored arc (probability fill)
            prob_angle = np.pi - prob * np.pi
            theta2 = np.linspace(np.pi, prob_angle, 200)
            arc_color = RED if prob >= 0.70 else ORANGE if prob >= user_thresh else GREEN
            ax.fill_between(np.cos(theta2), np.sin(theta2),
                             r_inner * np.sin(theta2), alpha=0.7, color=arc_color)
            ax.plot(np.cos(theta2), np.sin(theta2), color=arc_color, lw=3)
            ax.plot(r_inner*np.cos(theta2), r_inner*np.sin(theta2), color=arc_color, lw=3)

            # Needle
            needle_angle = np.pi - prob * np.pi
            ax.annotate("", xy=(0.72*np.cos(needle_angle), 0.72*np.sin(needle_angle)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2.5,
                                        mutation_scale=16))
            ax.plot(0, 0, "o", color=TEXT, markersize=8, zorder=5)

            # Labels
            ax.text(0, 0.22, f"{prob:.1%}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color=arc_color)
            ax.text(0, 0.06, "Churn Probability", ha="center", va="center",
                    fontsize=9, color=MUTED)
            ax.text(-0.92, 0.04, "0%",   ha="center", color=MUTED, fontsize=8)
            ax.text( 0.92, 0.04, "100%", ha="center", color=MUTED, fontsize=8)
            ax.text( 0,    0.85, label,  ha="center", color=arc_color,
                    fontsize=11, fontweight="bold")

            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.15, 1.1)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True); plt.close()

            # Risk card
            css_cls = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}[level]
            st.markdown(f"""
            <div style='background:#1e2130; border-radius:12px; padding:18px; margin-top:8px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;'>
                    <span style='color:#8b92a5; font-size:0.85rem;'>RISK LEVEL</span>
                    <span class='{css_cls}'>{label}</span>
                </div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                    <div style='background:#252836; border-radius:8px; padding:10px; text-align:center;'>
                        <div style='color:#8b92a5; font-size:0.7rem;'>PROBABILITY</div>
                        <div style='color:#e0e4f0; font-size:1.4rem; font-weight:700;'>{prob:.1%}</div>
                    </div>
                    <div style='background:#252836; border-radius:8px; padding:10px; text-align:center;'>
                        <div style='color:#8b92a5; font-size:0.7rem;'>THRESHOLD</div>
                        <div style='color:#e0e4f0; font-size:1.4rem; font-weight:700;'>{user_thresh:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key drivers for this customer
            st.markdown("#### Key Risk Factors")
            risk_factors = []
            if support >= 6:   risk_factors.append(("🔴", "High Support Calls",    f"{support} calls"))
            if delay >= 20:    risk_factors.append(("🔴", "High Payment Delay",    f"{delay} days"))
            if contract == "Monthly": risk_factors.append(("🟡", "Month-to-Month Contract", "Higher churn risk"))
            if tenure <= 12:   risk_factors.append(("🟡", "Low Tenure",            f"{tenure} months"))
            if usage <= 5:     risk_factors.append(("🟡", "Low Usage Frequency",   f"{usage} sessions"))
            if spend >= 800:   risk_factors.append(("🟢", "High Total Spend",      f"${spend}"))
            if contract == "Annual": risk_factors.append(("🟢", "Annual Contract",  "Lower churn risk"))
            if not risk_factors:
                risk_factors.append(("🟢", "Profile looks healthy", "No major risk signals"))

            for emoji, title, detail in risk_factors[:5]:
                st.markdown(f"""
                <div style='background:#1e2130; border-radius:8px; padding:10px 14px;
                             margin-bottom:6px; border-left:3px solid {"#ff4b4b" if emoji=="🔴" else "#ffa500" if emoji=="🟡" else "#00c853"};'>
                    <span style='font-size:0.85rem; color:#e0e4f0; font-weight:600;'>{emoji} {title}</span>
                    <span style='font-size:0.8rem; color:#8b92a5; margin-left:8px;'>{detail}</span>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: BATCH PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "📁  Batch Prediction":
    st.markdown("## 📁 Batch Customer Churn Prediction")
    st.caption("Upload a CSV of customers to score all at once — download results with risk labels.")

    col_up, col_info = st.columns([1.2, 1], gap="large")

    with col_info:
        st.markdown("#### Required CSV Columns")
        required = {
            "Column": ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls",
                       "Payment Delay", "Subscription Type", "Contract Length",
                       "Total Spend", "Last Interaction"],
            "Type":   ["int", "Male/Female", "int (months)", "int", "int",
                       "int (days)", "Basic/Standard/Premium", "Monthly/Quarterly/Annual",
                       "float ($)", "int (days)"],
        }
        st.dataframe(pd.DataFrame(required), use_container_width=True, hide_index=True)
        st.caption("CustomerID column is optional and will be preserved if present.")

        # Download sample template
        sample = test_df.drop(columns=["Churn"], errors="ignore").head(5)
        csv_bytes = sample.to_csv(index=False).encode()
        st.download_button("📥 Download Sample Template", csv_bytes,
                           "churn_template.csv", "text/csv")

    with col_up:
        st.markdown("#### Upload Customer Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if uploaded:
            input_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded **{len(input_df):,}** rows × **{len(input_df.columns)}** columns")

            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(input_df.head(10), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Scoring customers..."):
                    try:
                        results = predict_batch(input_df)
                        st.session_state["batch_results"] = results
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

    # Show results
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        st.markdown("---")
        st.markdown("### 📊 Batch Results")

        # Summary KPIs
        n_high   = (results["Risk_Level"] == "High").sum()
        n_medium = (results["Risk_Level"] == "Medium").sum()
        n_low    = (results["Risk_Level"] == "Low").sum()
        avg_prob = results["Churn_Probability"].mean()

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, color in [
            (k1, "Total Customers",  f"{len(results):,}",    "#6c63ff"),
            (k2, "High Risk",        f"{n_high:,}",           "#ff4b4b"),
            (k3, "Medium Risk",      f"{n_medium:,}",         "#ffa500"),
            (k4, "Avg Churn Prob",   f"{avg_prob:.1%}",       "#48c6ef"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='value' style='background:linear-gradient(135deg,{color},{color}aa);-webkit-background-clip:text;'>{value}</div>
                    <div class='label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)
        cr1, cr2 = st.columns([1.6, 1])

        with cr1:
            st.markdown("#### Prediction Table")
            display_cols = list(results.columns)
            styled = results.head(500)
            st.dataframe(styled, use_container_width=True, height=380)
            csv_out = results.to_csv(index=False).encode()
            st.download_button("📥 Download Full Results", csv_out,
                               "churn_predictions.csv", "text/csv",
                               use_container_width=True)

        with cr2:
            # Risk distribution pie
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
            sizes  = [n_high, n_medium, n_low]
            colors = [RED, ORANGE, GREEN]
            labels = [f"High\n{n_high:,}", f"Medium\n{n_medium:,}", f"Low\n{n_low:,}"]
            wedges, texts, auto = ax.pie(sizes, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.55, edgecolor=CARD_BG, linewidth=2))
            for t in texts: t.set_color(TEXT); t.set_fontsize(9)
            for t in auto:  t.set_color("white"); t.set_fontweight("bold")
            ax.set_title("Risk Distribution", color=TEXT, fontweight="bold")
            st.pyplot(fig, use_container_width=True); plt.close()

            # Probability histogram
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
            ax.hist(results["Churn_Probability"], bins=40, color=PURPLE, alpha=0.8, edgecolor=CARD_BG)
            ax.axvline(meta["best_threshold"], color=ORANGE, linestyle="--",
                       linewidth=1.5, label=f"Threshold ({meta['best_threshold']:.2f})")
            ax.set_xlabel("Churn Probability", color=MUTED)
            ax.set_ylabel("Count", color=MUTED)
            ax.set_title("Probability Distribution", color=TEXT, fontweight="bold")
            ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        # Use test set if nothing uploaded yet
    elif not uploaded:
        st.info("👆 Upload a CSV above, or use the **Download Sample Template** button to get started.")
        st.markdown("#### 🎲 Or try with the built-in test set")
        if st.button("▶ Score Built-in Test Set (64K customers)", use_container_width=False):
            with st.spinner("Scoring 64K customers..."):
                results = predict_batch(test_df.drop(columns=["Churn"], errors="ignore"))
                st.session_state["batch_results"] = results
            st.rerun()


# ════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":
    st.markdown("## 📈 Model Performance & Diagnostics")

    tab1, tab2, tab3 = st.tabs(["📉 ROC & PR Curves", "🎯 Threshold Analysis", "📋 Full Report"])

    # ── TAB 1: ROC & PR ──────────────────────────────────
    with tab1:
        col_roc, col_pr = st.columns(2)

        with col_roc:
            roc = meta["roc_curve"]
            fig, ax = plt.subplots(figsize=(6, 5.5))
            fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
            ax.plot(roc["fpr"], roc["tpr"], color=PURPLE, lw=2.5,
                    label=f"ROC Curve (AUC = {meta['test_roc_auc']:.4f})")
            ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.12, color=PURPLE)
            ax.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--", label="Random Classifier")
            ax.set_xlabel("False Positive Rate", color=MUTED)
            ax.set_ylabel("True Positive Rate", color=MUTED)
            ax.set_title("ROC Curve", color=TEXT, fontweight="bold", pad=12)
            ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=9)
            ax.set_xlim(0,1); ax.set_ylim(0,1.02)
            ax.grid(alpha=0.3)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

        with col_pr:
            pr = meta["pr_curve"]
            baseline = meta["churn_rate_test"]
            fig, ax = plt.subplots(figsize=(6, 5.5))
            fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
            ax.plot(pr["recall"], pr["precision"], color=CYAN, lw=2.5,
                    label=f"PR Curve (AUC = {meta['test_pr_auc']:.4f})")
            ax.fill_between(pr["recall"], pr["precision"], alpha=0.12, color=CYAN)
            ax.axhline(baseline, color=MUTED, lw=1, linestyle="--",
                       label=f"Baseline (prevalence = {baseline:.2f})")
            ax.set_xlabel("Recall", color=MUTED)
            ax.set_ylabel("Precision", color=MUTED)
            ax.set_title("Precision-Recall Curve", color=TEXT, fontweight="bold", pad=12)
            ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=9)
            ax.set_xlim(0,1); ax.set_ylim(0,1.02)
            ax.grid(alpha=0.3)
            ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True); plt.close()

    # ── TAB 2: Threshold Analysis ─────────────────────────
    with tab2:
        sweep = pd.DataFrame(meta["threshold_sweep"])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor(CARD_BG); ax.set_facecolor(CARD_BG)
        ax.plot(sweep["threshold"], sweep["f1"],        color=PURPLE, lw=2.5, label="F1",        marker="o", ms=4)
        ax.plot(sweep["threshold"], sweep["precision"],  color=CYAN,   lw=2.5, label="Precision", marker="s", ms=4)
        ax.plot(sweep["threshold"], sweep["recall"],     color=RED,    lw=2.5, label="Recall",    marker="^", ms=4)
        ax.plot(sweep["threshold"], sweep["accuracy"],   color=GREEN,  lw=2.5, label="Accuracy",  marker="D", ms=4)
        ax.axvline(meta["best_threshold"], color=ORANGE, lw=2, linestyle="--",
                   label=f"Optimal Threshold = {meta['best_threshold']:.3f}")
        ax.set_xlabel("Decision Threshold", color=MUTED)
        ax.set_ylabel("Score", color=MUTED)
        ax.set_title("Metrics vs Decision Threshold", color=TEXT, fontweight="bold", pad=12)
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT, fontsize=9, ncol=3)
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True); plt.close()

        st.dataframe(sweep.rename(columns=str.title).style.highlight_max(
            subset=["F1","Accuracy"], color="#6c63ff33").format("{:.4f}", subset=["F1","Precision","Recall","Accuracy"]),
            use_container_width=True, hide_index=True)

    # ── TAB 3: Full Report ───────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📋 Model Card")
            report_data = {
                "Property":  ["Algorithm", "Training Rows", "Test Rows",
                               "Features", "Optimal Threshold",
                               "Accuracy", "ROC-AUC", "PR-AUC",
                               "F1 Score", "Precision", "Recall"],
                "Value":     [meta["model_name"], f"{meta['train_size']:,}", f"{meta['test_size']:,}",
                               str(len(meta["feature_names"])), f"{meta['best_threshold']:.3f}",
                               f"{meta['test_accuracy']:.4f}", f"{meta['test_roc_auc']:.4f}",
                               f"{meta['test_pr_auc']:.4f}", f"{meta['test_f1']:.4f}",
                               f"{meta['test_precision']:.4f}", f"{meta['test_recall']:.4f}"],
            }
            st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### 🔬 Feature List")
            feat_data = pd.DataFrame(meta["feature_importances"])
            feat_data["importance"] = feat_data["importance"].round(4)
            feat_data.index = feat_data.index + 1
            st.dataframe(feat_data, use_container_width=True, height=380)

        st.markdown("#### ⚙️ Pipeline Architecture")
        st.code("""
CustomerChurnPipeline
│
├── 1. Data Ingestion
│      └── CSV load → dropna() → reset_index
│
├── 2. Feature Engineering
│      ├── SpendPerTenure    = Total Spend / (Tenure + 1)
│      ├── CallsPerTenure    = Support Calls / (Tenure + 1)
│      ├── DelayRatio        = Payment Delay / (Tenure + 1)
│      ├── EngagementScore   = Usage Frequency × Tenure
│      ├── SupportIntensity  = Support Calls × Payment Delay
│      ├── HighRiskFlag      = (Calls > 5) AND (Delay > 15)
│      ├── AgeGroup          = pd.cut(Age, 5 bins)
│      └── TenureGroup       = pd.cut(Tenure, 5 bins)
│
├── 3. Preprocessing (sklearn ColumnTransformer)
│      ├── Numeric  → StandardScaler (13 features)
│      └── Categorical → OrdinalEncoder (5 features)
│
├── 4. Model — Logistic Regression
│      ├── C=0.5, class_weight='balanced'
│      ├── Trained on 100K stratified sample
│      └── Calibrated probabilities (logistic output)
│
└── 5. Threshold Tuning
       └── Optimal threshold = {:.3f} (val F1-maximized)
        """.format(meta["best_threshold"]), language="text")
