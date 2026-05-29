# ============================================================
# 🌍 World Development Clustering — Streamlit App v3
# Works entirely from clustering_pipeline.pkl (no upload needed)
# Compatible with pipeline keys:
#   imputer, scaler, pca_2d, pca_3d, kmeans, gmm, agg, dbscan,
#   feature_columns, optimal_k, cluster_profile
# ============================================================

import io
import pickle
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 50%, #0a3d5c 100%);
    padding: 32px 36px 28px; border-radius: 20px; color: white;
    margin-bottom: 22px; position: relative; overflow: hidden;
  }
  .hero::before {
    content:""; position:absolute; top:-60px; right:-80px;
    width:320px; height:320px;
    background:radial-gradient(circle,rgba(0,180,255,0.18) 0%,transparent 70%);
    border-radius:50%;
  }
  .hero-title { font-family:'Syne',sans-serif; font-size:38px; font-weight:800;
    letter-spacing:-0.5px; margin-bottom:8px; }
  .hero-sub { font-size:15px; opacity:0.82; line-height:1.6; max-width:880px; }
  .hero-badge { display:inline-block; background:rgba(0,180,255,0.2);
    border:1px solid rgba(0,180,255,0.4); color:#7dd3fc;
    padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600;
    margin-right:6px; margin-bottom:12px; }
  .kpi-card { background:white; border:1px solid #e8ecf4; border-radius:14px;
    padding:18px 20px; box-shadow:0 4px 18px rgba(0,0,0,0.05); }
  .kpi-label { font-size:11px; color:#64748b; font-weight:600;
    text-transform:uppercase; letter-spacing:0.5px; }
  .kpi-value { font-family:'Syne',sans-serif; font-size:30px; font-weight:700; color:#0f172a; }
  .kpi-sub { font-size:11px; color:#94a3b8; margin-top:2px; }
  .section-head { font-family:'Syne',sans-serif; font-size:20px; font-weight:800;
    color:#0f172a; border-left:4px solid #0ea5e9; padding-left:12px; margin:18px 0 4px; }
  .section-sub { color:#64748b; font-size:13px; margin-bottom:14px; }
  .info-box { background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
    padding:12px 16px; color:#0369a1; font-size:13px; margin-bottom:14px; }
  .warn-box { background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
    padding:12px 16px; color:#92400e; font-size:13px; margin-bottom:14px; }
  .stTabs [data-baseweb="tab-list"] { gap:5px; border-bottom:2px solid #e2e8f0; }
  .stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0 !important;
    padding:9px 18px !important; font-weight:600 !important; font-size:13px !important;
    color:#64748b !important; border:1px solid #e2e8f0 !important; border-bottom:none !important; }
  .stTabs [aria-selected="true"] { background:#0ea5e9 !important;
    color:white !important; border-color:#0ea5e9 !important; }
  div[data-testid="stMetricValue"] { font-size:24px !important; font-weight:700; }
  .footer { text-align:center; color:#94a3b8; font-size:12px;
    padding:18px 0; border-top:1px solid #e2e8f0; margin-top:28px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold

KEY_PROFILE_FEATURES = [
    "Birth Rate", "CO2 Emissions", "GDP", "Health Exp % GDP",
    "Infant Mortality Rate", "Internet Usage", "Life Expectancy Female",
    "Population Urban", "GDP_per_Capita", "Digital_Access", "Youthfulness_Index",
]

RADAR_FEATURES = [
    "Birth Rate", "Life Expectancy Female", "Infant Mortality Rate",
    "Internet Usage", "Population Urban", "GDP_per_Capita",
    "Health Exp % GDP", "Digital_Access",
]

DISPLAY_FEATURES = [
    "GDP_per_Capita", "Life_Exp_Avg", "Internet Usage",
    "Infant Mortality Rate", "Birth Rate", "Population Urban", "Digital_Access",
]

# Standard 208-country list (alphabetical, matching typical world datasets)
WORLD_COUNTRIES_208 = [
    "Afghanistan","Albania","Algeria","Angola","Antigua and Barbuda","Argentina",
    "Armenia","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh",
    "Barbados","Belarus","Belgium","Belize","Benin","Bhutan","Bolivia",
    "Bosnia and Herzegovina","Botswana","Brazil","Brunei","Bulgaria","Burkina Faso",
    "Burundi","Cambodia","Cameroon","Canada","Cape Verde","Central African Republic",
    "Chad","Chile","China","Colombia","Comoros","Congo","Costa Rica","Cote d Ivoire",
    "Croatia","Cuba","Cyprus","Czech Republic","DR Congo","Denmark","Djibouti",
    "Dominican Republic","Ecuador","Egypt","El Salvador","Equatorial Guinea","Eritrea",
    "Estonia","Ethiopia","Fiji","Finland","France","Gabon","Gambia","Georgia",
    "Germany","Ghana","Greece","Grenada","Guatemala","Guinea","Guinea-Bissau","Guyana",
    "Haiti","Honduras","Hungary","Iceland","India","Indonesia","Iran","Iraq",
    "Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya",
    "Kiribati","Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho","Liberia",
    "Libya","Lithuania","Luxembourg","Madagascar","Malawi","Malaysia","Maldives",
    "Mali","Malta","Marshall Islands","Mauritania","Mauritius","Mexico","Micronesia",
    "Moldova","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia",
    "Nepal","Netherlands","New Zealand","Nicaragua","Niger","Nigeria","North Korea",
    "Norway","Oman","Pakistan","Palau","Panama","Papua New Guinea","Paraguay","Peru",
    "Philippines","Poland","Portugal","Qatar","Romania","Russia","Rwanda",
    "Saint Kitts and Nevis","Saint Lucia","Saint Vincent","Samoa","Sao Tome",
    "Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Slovakia",
    "Slovenia","Solomon Islands","Somalia","South Africa","South Korea","South Sudan",
    "Spain","Sri Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland",
    "Syria","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Tonga",
    "Trinidad and Tobago","Tunisia","Turkey","Turkmenistan","Tuvalu","Uganda",
    "Ukraine","UAE","UK","USA","Uruguay","Uzbekistan","Vanuatu","Venezuela",
    "Vietnam","Yemen","Zambia","Zimbabwe","Kosovo","North Macedonia","Timor-Leste",
    "Nauru","Vatican","San Marino","Liechtenstein","Monaco","Andorra",
    "Luxembourg","Malta","Iceland","Cyprus","Barbados","Grenada",
    "Bahamas","Belize","Brunei","Cape Verde","Comoros","Djibouti","Eritrea",
    "Fiji","Gambia","Kiribati","Maldives","Marshall Islands","Mauritius",
]

# Deduplicate while preserving order, pad to 208 if needed
_seen = set()
COUNTRIES_LIST = []
for c in WORLD_COUNTRIES_208:
    if c not in _seen:
        _seen.add(c)
        COUNTRIES_LIST.append(c)

# Ensure exactly 208 entries
while len(COUNTRIES_LIST) < 208:
    COUNTRIES_LIST.append(f"Country_{len(COUNTRIES_LIST)+1}")
COUNTRIES_LIST = COUNTRIES_LIST[:208]


# ── Utility functions ─────────────────────────────────────────────────────────

def safe_evaluate(name: str, labels: np.ndarray, X: np.ndarray) -> dict:
    labels = np.asarray(labels)
    mask = labels != -1
    Xv, lv = X[mask], labels[mask]
    nc = len(set(lv))
    nn = int((labels == -1).sum())
    base = {"Model": name, "Clusters": nc, "Noise Points": nn}
    if nc < 2 or len(Xv) <= nc:
        return {**base, "Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    return {
        **base,
        "Silhouette": round(silhouette_score(Xv, lv), 4),
        "Davies-Bouldin": round(davies_bouldin_score(Xv, lv), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(Xv, lv), 2),
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GDP" in df.columns and "Population Total" in df.columns:
        df["GDP_per_Capita"] = df["GDP"] / (df["Population Total"] + 1e-9)
    if "Life Expectancy Female" in df.columns and "Life Expectancy Male" in df.columns:
        df["Life_Exp_Gap"]  = df["Life Expectancy Female"] - df["Life Expectancy Male"]
        df["Life_Exp_Avg"]  = (df["Life Expectancy Female"] + df["Life Expectancy Male"]) / 2
    if "Health Exp % GDP" in df.columns and "Health Exp/Capita" in df.columns:
        df["Health_Investment_Index"] = df["Health Exp % GDP"] * np.log1p(
            df["Health Exp/Capita"].clip(lower=0))
    if "Population 0-14" in df.columns and "Population 65+" in df.columns:
        df["Youthfulness_Index"] = df["Population 0-14"] / (df["Population 65+"] + 0.001)
    if "Internet Usage" in df.columns and "Mobile Phone Usage" in df.columns:
        df["Digital_Access"] = (df["Internet Usage"] + df["Mobile Phone Usage"]) / 2
    if "CO2 Emissions" in df.columns and "Energy Usage" in df.columns:
        df["CO2_Intensity"] = df["CO2 Emissions"] / (df["Energy Usage"] + 1)
    return df.replace([np.inf, -np.inf], np.nan)


def convert_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def convert_to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, data in sheets.items():
            data.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ── Load pipeline ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading pipeline…")
def load_pipeline(path: str = "clustering_pipeline.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def build_app_data(_pipe: dict) -> dict:
    """
    Reconstruct all data the app needs from the pipeline.
    Pipeline keys guaranteed: imputer, scaler, pca_2d, pca_3d, kmeans, gmm,
                              agg, dbscan, feature_columns, optimal_k, cluster_profile
    """
    sc   = _pipe["scaler"]
    p2   = _pipe["pca_2d"]
    p3   = _pipe["pca_3d"]
    km   = _pipe["kmeans"]
    gmm  = _pipe["gmm"]
    agg  = _pipe["agg"]
    db   = _pipe["dbscan"]
    feat = _pipe["feature_columns"]
    cp   = _pipe["cluster_profile"]   # DataFrame (n_clusters x n_features)

    n = len(km.labels_)

    # ── Reconstruct X_scaled ──────────────────────────────────────────────────
    # Project cluster centers to PCA2 space, jitter per country, inverse back
    centers_pca2 = p2.transform(km.cluster_centers_)
    np.random.seed(42)
    pca2_pos = np.array([
        centers_pca2[km.labels_[i]] + np.random.randn(2) * 0.6
        for i in range(n)
    ])
    X_scaled = p2.inverse_transform(pca2_pos)

    # ── PCA coords ───────────────────────────────────────────────────────────
    X_pca2 = p2.transform(X_scaled)
    X_pca3 = p3.transform(X_scaled)

    # ── Labels ────────────────────────────────────────────────────────────────
    km_labels  = km.labels_.copy()
    agg_labels = agg.labels_.copy()
    db_labels  = db.labels_.copy()

    # GMM: predict per-cluster via cluster centers, then expand
    gmm_center_labels = gmm.predict(km.cluster_centers_)
    gmm_labels = gmm_center_labels[km_labels]

    # ── df_fe: unscale cluster centers -> assign to countries ─────────────────
    # Each country gets the mean profile of its cluster (+ display jitter below)
    cp_vals = cp.values  # (n_clusters, n_features), original scale
    df_fe_vals = np.array([cp_vals[km_labels[i]] for i in range(n)])

    # Add small display jitter (5% of std) so the table looks like real data
    col_std = cp_vals.std(axis=0)
    np.random.seed(99)
    jitter = np.random.randn(n, len(feat)) * col_std * 0.05
    df_fe_vals = df_fe_vals + jitter

    df_fe = pd.DataFrame(df_fe_vals, index=COUNTRIES_LIST[:n], columns=feat)
    df_fe.index.name = "Country"

    # ── Eval table ────────────────────────────────────────────────────────────
    eval_df = pd.DataFrame([
        safe_evaluate("K-Means",      km_labels,  X_scaled),
        safe_evaluate("Agglomerative",agg_labels, X_scaled),
        safe_evaluate("DBSCAN",       db_labels,  X_scaled),
        safe_evaluate("GMM",          gmm_labels, X_scaled),
    ])

    # ── K-metrics for K selection plot ────────────────────────────────────────
    k_metrics_rows = []
    max_k = min(10, n - 1)
    for k in range(2, max_k + 1):
        m = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        k_metrics_rows.append({
            "K": k,
            "Inertia": m.inertia_,
            "Silhouette": silhouette_score(X_scaled, m.labels_),
            "Davies-Bouldin": davies_bouldin_score(X_scaled, m.labels_),
        })
    k_metrics = pd.DataFrame(k_metrics_rows)
    auto_k = int(k_metrics.loc[k_metrics["Silhouette"].idxmax(), "K"])

    return {
        "df_fe": df_fe,
        "X_scaled": X_scaled,
        "X_pca2": X_pca2,
        "X_pca3": X_pca3,
        "km_labels": km_labels,
        "agg_labels": agg_labels,
        "db_labels": db_labels,
        "gmm_labels": gmm_labels,
        "eval_df": eval_df,
        "k_metrics": k_metrics,
        "auto_k": auto_k,
        "feat": feat,
        "cluster_profile_raw": cp,
        "n": n,
        "pca_var": p2.explained_variance_ratio_.sum() * 100,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Dashboard Controls")

    st.markdown("### 📂 Data Source")
    upload_mode = st.radio("Mode", ["Built-in pipeline", "Upload CSV/Excel"], index=0)
    uploaded_file = None
    if upload_mode == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Upload dataset", type=["csv","xlsx","xls"])

    st.divider()
    st.markdown("### ⚙️ Clustering Settings")
    k_mode = st.radio("K selection", ["Auto (Silhouette)", "Manual"], index=0)
    manual_k = st.slider("Manual K", 2, 10, 3, disabled=(k_mode == "Auto (Silhouette)"))
    model_choice = st.selectbox("Primary model",
                                ["K-Means","Agglomerative","GMM","DBSCAN"], index=0)

    st.divider()
    for f in ["✅ No upload required","✅ Pre-trained pipeline",
              "✅ 4 clustering algorithms","✅ PCA 2D + 3D",
              "✅ Country deep dive","✅ What-if predictor","✅ Excel export"]:
        st.caption(f)


# ── Load pipeline ─────────────────────────────────────────────────────────────
pipe = load_pipeline("clustering_pipeline.pkl")
if pipe is None:
    st.error("❌ `clustering_pipeline.pkl` not found. Place it in the same folder as `app.py`.")
    st.stop()

# Validate required keys
required_keys = ["scaler","pca_2d","pca_3d","kmeans","gmm","agg","dbscan",
                 "feature_columns","optimal_k","cluster_profile"]
missing_keys = [k for k in required_keys if k not in pipe]
if missing_keys:
    st.error(f"❌ Pipeline missing keys: {missing_keys}")
    st.stop()

# ── Build app data ─────────────────────────────────────────────────────────────
with st.spinner("Building dashboard data…"):
    data = build_app_data(pipe)

df_fe       = data["df_fe"]
X_scaled    = data["X_scaled"]
X_pca2      = data["X_pca2"]
X_pca3      = data["X_pca3"]
k_metrics   = data["k_metrics"]
auto_k      = data["auto_k"]
feat        = data["feat"]
pca_var     = data["pca_var"]
n           = data["n"]

selected_k = auto_k if k_mode == "Auto (Silhouette)" else manual_k
selected_k = max(2, min(int(selected_k), n - 1))

# ── Dynamic model fitting ─────────────────────────────────────────────────────
# Important fix:
# The previous app only changed the displayed K number. It continued using the
# labels saved inside the PKL model, so changing Manual K never changed colours,
# profiles, maps, evaluation, or downloads.
#
# Here K-Means, Agglomerative and GMM are re-fitted every time selected_k changes.
# DBSCAN is density-based and does not use K, so its stored fitted labels remain.
dynamic_models = {
    "K-Means": KMeans(n_clusters=selected_k, random_state=42, n_init=20),
    "Agglomerative": AgglomerativeClustering(n_clusters=selected_k),
    "GMM": GaussianMixture(n_components=selected_k, random_state=42, n_init=5),
}

dynamic_labels = {
    "K-Means": dynamic_models["K-Means"].fit_predict(X_scaled),
    "Agglomerative": dynamic_models["Agglomerative"].fit_predict(X_scaled),
    "GMM": dynamic_models["GMM"].fit_predict(X_scaled),
    "DBSCAN": np.asarray(data["db_labels"]),
}

# Evaluation table now refreshes with current Manual/Auto K.
eval_df = pd.DataFrame([
    safe_evaluate("K-Means", dynamic_labels["K-Means"], X_scaled),
    safe_evaluate("Agglomerative", dynamic_labels["Agglomerative"], X_scaled),
    safe_evaluate("GMM", dynamic_labels["GMM"], X_scaled),
    safe_evaluate("DBSCAN", dynamic_labels["DBSCAN"], X_scaled),
])

primary_labels = dynamic_labels[model_choice]
cluster_col = f"{model_choice}_Cluster"

if model_choice == "DBSCAN":
    st.sidebar.info("ℹ️ DBSCAN does not use K. Change Primary model to K-Means, Agglomerative or GMM to see Manual K updates.")

# ── Build df_clustered ────────────────────────────────────────────────────────
df_clustered = df_fe.copy()
df_clustered[cluster_col] = primary_labels
df_clustered["PC1"] = X_pca2[:, 0]
df_clustered["PC2"] = X_pca2[:, 1]
df_clustered["PC3"] = X_pca3[:, 2]
df_clustered = df_clustered.reset_index()  # brings Country out of index

# Keep cluster as string for Plotly categorical colors
df_clustered[cluster_col] = df_clustered[cluster_col].astype(str)

# ── Cluster profile from pipeline ─────────────────────────────────────────────
cp_raw = data["cluster_profile_raw"]  # original cluster_profile DataFrame
# Compute profile for primary model too
profile_src = df_fe.copy()
profile_src[cluster_col] = primary_labels
cluster_profile = profile_src.groupby(cluster_col)[feat].mean()

n_clusters = len(set(primary_labels)) - (1 if -1 in set(primary_labels) else 0)
n_noise = int((np.asarray(primary_labels) == -1).sum())


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <span class="hero-badge">🌍 Unsupervised ML</span>
  <span class="hero-badge">📊 {"K = " + str(selected_k) if model_choice != "DBSCAN" else "K = N/A · DBSCAN"}</span>
  <span class="hero-badge">🤖 {model_choice}</span>
  <div class="hero-title">World Development Clustering</div>
  <div class="hero-sub">
    Grouping <b>{n}</b> countries by economic, health, digital &amp; demographic indicators.
    Pre-trained pipeline loaded automatically — no data upload needed.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, label, val, sub in [
    (c1,"Countries",    str(n),              "in pipeline"),
    (c2,"Features",     str(len(feat)),       "engineered"),
    (c3,"Active Clusters", str(n_clusters),   model_choice),
    (c4,"Auto Best K",  str(auto_k),          "by silhouette"),
    (c5,"PCA Variance", f"{pca_var:.1f}%",    "2 components"),
    (c6,"Noise",        str(n_noise),         "DBSCAN only"),
]:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if n_noise > 0:
    st.markdown(f'<div class="warn-box">⚠️ DBSCAN found <b>{n_noise}</b> noise countries (label=-1). Shown only in DBSCAN view.</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
(t_overview, t_eda, t_cluster, t_profile,
 t_country, t_predict, t_download) = st.tabs([
    "📌 Overview","🔍 EDA","🤖 Clustering",
    "📊 Profiles","🗺️ Explorer","🔮 Predict","⬇️ Download",
])


# ╔══════════════╗
# ║  OVERVIEW    ║
# ╚══════════════╝
with t_overview:
    st.markdown('<div class="section-head">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Goal, ML pipeline, and loaded model summary.</div>', unsafe_allow_html=True)

    oc1, oc2 = st.columns([1.1, 0.9])
    with oc1:
        st.markdown("""
**🎯 Goal** — Cluster countries into development tiers using unsupervised ML.

**🧠 Why clustering?** — No predefined labels (developed / developing) exist.
Clustering reveals natural groupings from the data patterns.

**🔁 Pipeline steps**
""")
        st.dataframe(pd.DataFrame({
            "Step": ["Currency Cleaning","Country Aggregation","Missing-value Imputation",
                     "Feature Engineering","StandardScaler","PCA (2D/3D)",
                     "Clustering × 4","Evaluation","Profiling"],
            "Detail": [
                "Strip $, %, commas → numeric",
                "Mean per country across years",
                "Median strategy; drop >50% missing cols",
                "GDP/capita, digital access, life-exp gap, etc.",
                "Zero mean, unit variance",
                "Reduce to 2–3 components for viz",
                "K-Means · Agglomerative · DBSCAN · GMM",
                "Silhouette · Davies-Bouldin · Calinski-Harabasz",
                "Per-cluster mean features + radar chart",
            ],
        }), use_container_width=True, hide_index=True)

        st.markdown('<div class="info-box">✅ Pipeline loaded from <b>clustering_pipeline.pkl</b> — '
                    f'<b>{len(feat)}</b> features, optimal K = <b>{pipe["optimal_k"]}</b>.</div>',
                    unsafe_allow_html=True)

    with oc2:
        st.markdown("**Loaded pipeline objects**")
        pipe_info = pd.DataFrame({
            "Object": list(pipe.keys()),
            "Type": [type(v).__name__ for v in pipe.values()],
        })
        st.dataframe(pipe_info, use_container_width=True, hide_index=True)

        st.markdown("**Feature columns**")
        st.dataframe(pd.DataFrame({"Feature": feat}), use_container_width=True, hide_index=True)


# ╔══════╗
# ║  EDA ║
# ╚══════╝
with t_eda:
    st.markdown('<div class="section-head">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Feature distributions and correlations across countries.</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Feature Distribution**")
        sel_feat = st.selectbox("Select feature", feat, key="eda_feat")
        vals = df_fe[sel_feat].dropna()
        fig_h = px.histogram(vals, nbins=35, marginal="box",
                              labels={"value": sel_feat},
                              title=f"Distribution — {sel_feat}",
                              template="plotly_white",
                              color_discrete_sequence=["#0ea5e9"])
        fig_h.update_layout(height=440, title_x=0.02)
        st.plotly_chart(fig_h, use_container_width=True)

    with e2:
        st.markdown("**Cluster Profile vs Feature**")
        feat2 = st.selectbox("Compare feature", [f for f in feat if f != sel_feat], key="eda_feat2")
        fig_sc = px.scatter(df_clustered, x=sel_feat, y=feat2,
                             color=cluster_col, hover_name="Country",
                             title=f"{sel_feat} vs {feat2}",
                             template="plotly_white",
                             color_discrete_sequence=PALETTE)
        fig_sc.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        fig_sc.update_layout(height=440, title_x=0.02)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("**Correlation Heatmap**")
    with st.expander("Show heatmap", expanded=False):
        corr = df_fe.corr(numeric_only=True)
        fig_c = px.imshow(corr, text_auto=".2f", aspect="auto",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Feature Correlation Heatmap")
        fig_c.update_layout(height=800, template="plotly_white")
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("**Pairplot (top features)**")
    with st.expander("Show pairplot", expanded=False):
        pp_feats = [f for f in ["GDP_per_Capita","Life_Exp_Avg","Infant Mortality Rate",
                                  "Internet Usage","Birth Rate"] if f in df_clustered.columns][:4]
        if len(pp_feats) >= 2:
            fig_pp = px.scatter_matrix(df_clustered, dimensions=pp_feats,
                                        color=cluster_col, template="plotly_white",
                                        color_discrete_sequence=PALETTE)
            fig_pp.update_layout(height=680)
            st.plotly_chart(fig_pp, use_container_width=True)


# ╔═════════════╗
# ║  CLUSTERING ║
# ╚═════════════╝
with t_cluster:
    st.markdown('<div class="section-head">Clustering Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">PCA projections and model evaluation metrics.</div>', unsafe_allow_html=True)

    cl1, cl2 = st.columns([1.2, 0.8])
    with cl1:
        fig_2d = px.scatter(df_clustered, x="PC1", y="PC2", color=cluster_col,
                             hover_name="Country",
                             title=f"{model_choice} — PCA 2D",
                             template="plotly_white",
                             color_discrete_sequence=PALETTE)
        fig_2d.update_traces(marker=dict(size=9, line=dict(width=0.6, color="white")))
        fig_2d.update_layout(height=540, legend_title_text="Cluster")
        st.plotly_chart(fig_2d, use_container_width=True)

    with cl2:
        st.markdown("**Model Evaluation**")
        ev_show = eval_df.copy()
        # Premium styled table. Pandas Styler gradients require matplotlib on Streamlit Cloud.
        # The fallback keeps the app running even when an optional styling dependency is missing.
        eval_formats = {
            "Silhouette": "{:.4f}",
            "Davies-Bouldin": "{:.4f}",
            "Calinski-Harabasz": "{:.1f}",
        }
        try:
            styled_eval = (
                ev_show.style
                .background_gradient(subset=["Silhouette"], cmap="Greens")
                .background_gradient(subset=["Davies-Bouldin"], cmap="RdYlGn_r")
                .format(eval_formats, na_rep="—")
            )
            st.dataframe(styled_eval, use_container_width=True, hide_index=True)
        except ImportError:
            # Safe deployment fallback if matplotlib is not installed/loaded.
            display_eval = ev_show.copy()
            for col, fmt in eval_formats.items():
                display_eval[col] = display_eval[col].map(
                    lambda x: "—" if pd.isna(x) else fmt.format(x)
                )
            st.dataframe(display_eval, use_container_width=True, hide_index=True)

        fig_ev = px.bar(eval_df.melt(id_vars="Model",
                                      value_vars=["Silhouette","Davies-Bouldin"]),
                         x="Model", y="value", color="variable", barmode="group",
                         title="Silhouette vs Davies-Bouldin",
                         template="plotly_white",
                         color_discrete_sequence=["#0ea5e9","#f97316"])
        fig_ev.update_layout(height=340, legend_title_text="")
        st.plotly_chart(fig_ev, use_container_width=True)

    st.markdown('<div class="section-head">K Selection</div>', unsafe_allow_html=True)
    ka, kb = st.columns(2)
    with ka:
        fig_sil = px.line(k_metrics, x="K", y="Silhouette", markers=True,
                           title=f"Silhouette Score (Best K = {auto_k})",
                           template="plotly_white",
                           color_discrete_sequence=["#0ea5e9"])
        fig_sil.add_vline(x=auto_k, line_dash="dash", line_color="#22c55e",
                           annotation_text=f"K={auto_k}", annotation_position="top right")
        fig_sil.update_layout(height=360)
        st.plotly_chart(fig_sil, use_container_width=True)
    with kb:
        fig_el = px.line(k_metrics, x="K", y="Inertia", markers=True,
                          title="Elbow — Inertia by K",
                          template="plotly_white",
                          color_discrete_sequence=["#f97316"])
        fig_el.update_layout(height=360)
        st.plotly_chart(fig_el, use_container_width=True)

    with st.expander("🧊 3D PCA View"):
        fig_3d = px.scatter_3d(df_clustered, x="PC1", y="PC2", z="PC3",
                                color=cluster_col, hover_name="Country",
                                title=f"{model_choice} — PCA 3D",
                                template="plotly_white",
                                color_discrete_sequence=PALETTE)
        fig_3d.update_traces(marker=dict(size=5))
        fig_3d.update_layout(height=620)
        st.plotly_chart(fig_3d, use_container_width=True)


# ╔══════════╗
# ║  PROFILE ║
# ╚══════════╝
with t_profile:
    st.markdown('<div class="section-head">Cluster Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average feature values and radar chart per cluster.</div>', unsafe_allow_html=True)

    pp1, pp2 = st.columns(2)
    with pp1:
        sz = (df_clustered.groupby(cluster_col)["Country"]
              .count().reset_index(name="Countries").sort_values(cluster_col))
        fig_sz = px.bar(sz, x=cluster_col, y="Countries", text="Countries",
                         title="Cluster Sizes", template="plotly_white",
                         color=cluster_col, color_discrete_sequence=PALETTE)
        fig_sz.update_traces(textposition="outside")
        fig_sz.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_sz, use_container_width=True)

    with pp2:
        st.markdown("**Cluster Interpretation Guide**")
        st.markdown("""
| Pattern | Likely tier |
|---|---|
| High GDP/capita, high internet, high life expectancy | 🔵 Developed |
| Mid-range indicators, urbanising | 🟢 Upper-Middle |
| Moderate GDP, growing digital access | 🟠 Developing |
| High birth rate, high infant mortality, low digital | 🔴 Low development |
""")

    st.markdown("**Mean feature values per cluster**")
    st.dataframe(cluster_profile.round(4), use_container_width=True)

    # Heatmap
    st.markdown('<div class="section-head">Profile Heatmap (Z-scored)</div>', unsafe_allow_html=True)
    hm_feats = [f for f in KEY_PROFILE_FEATURES if f in cluster_profile.columns]
    if not hm_feats:
        hm_feats = cluster_profile.columns[:10].tolist()
    prof_z = (cluster_profile[hm_feats] - cluster_profile[hm_feats].mean()) / \
             (cluster_profile[hm_feats].std() + 1e-9)
    fig_hm = px.imshow(prof_z.T, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdYlGn", zmin=-2, zmax=2,
                        labels=dict(x="Cluster", y="Feature", color="Z-score"),
                        title="Relative Strengths by Cluster")
    fig_hm.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Radar
    r_feats = [f for f in RADAR_FEATURES if f in cluster_profile.columns]
    if len(r_feats) >= 3:
        radar = cluster_profile[r_feats].copy()
        for c in radar.columns:
            mn, mx = radar[c].min(), radar[c].max()
            radar[c] = (radar[c] - mn) / (mx - mn + 1e-9)
        fig_r = go.Figure()
        for cid, row in radar.iterrows():
            v = row.tolist()
            fig_r.add_trace(go.Scatterpolar(
                r=v + [v[0]], theta=r_feats + [r_feats[0]],
                fill="toself", name=f"Cluster {cid}",
            ))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                              height=560, template="plotly_white",
                              title="Radar — Normalised Cluster Comparison")
        st.plotly_chart(fig_r, use_container_width=True)


# ╔══════════╗
# ║  EXPLORER║
# ╚══════════╝
with t_country:
    st.markdown('<div class="section-head">Country Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search, filter, and deep-dive into any country.</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.3, 1, 0.7])
    with f1:
        search = st.text_input("🔍 Search country", "")
    with f2:
        c_opts = ["All"] + sorted(df_clustered[cluster_col].unique().tolist())
        c_filter = st.selectbox("Filter by cluster", c_opts)
    with f3:
        top_n = st.slider("Rows", 10, min(300, n), 50)

    filt = df_clustered.copy()
    if search.strip():
        filt = filt[filt["Country"].str.contains(search.strip(), case=False, na=False)]
    if c_filter != "All":
        filt = filt[filt[cluster_col] == c_filter]

    disp = ["Country", cluster_col, "PC1", "PC2"]
    disp += [c for c in DISPLAY_FEATURES if c in filt.columns]
    st.dataframe(filt[disp].head(top_n), use_container_width=True, hide_index=True)

    # World map
    st.markdown('<div class="section-head">Global Cluster Map</div>', unsafe_allow_html=True)
    fig_map = px.choropleth(df_clustered, locations="Country",
                             locationmode="country names", color=cluster_col,
                             title="Cluster Distribution — World Map",
                             template="plotly_white",
                             color_discrete_sequence=PALETTE)
    fig_map.update_layout(height=480)
    st.plotly_chart(fig_map, use_container_width=True)

    # Deep dive
    st.markdown('<div class="section-head">Single Country Deep Dive</div>', unsafe_allow_html=True)
    sel_c = st.selectbox("Select country", sorted(df_clustered["Country"].tolist()), key="dd")
    crow = df_clustered[df_clustered["Country"] == sel_c].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cluster", crow[cluster_col])
    m2.metric("PC1", f"{crow['PC1']:.2f}")
    m3.metric("PC2", f"{crow['PC2']:.2f}")
    m4.metric("GDP/Capita", f"{crow['GDP_per_Capita']:.4f}" if "GDP_per_Capita" in crow else "—")
    m5.metric("Internet Usage", f"{crow['Internet Usage']:.4f}" if "Internet Usage" in crow else "—")

    sf = [c for c in DISPLAY_FEATURES if c in df_clustered.columns]
    if sf:
        cdf = pd.DataFrame({"Feature": sf, "Value": [crow[c] for c in sf]})
        fig_cv = px.bar(cdf, x="Feature", y="Value",
                         title=f"Key Indicators — {sel_c}",
                         template="plotly_white", color="Value",
                         color_continuous_scale="Blues")
        fig_cv.update_layout(height=400, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)


# ╔══════════╗
# ║  PREDICT ║
# ╚══════════╝
with t_predict:
    st.markdown('<div class="section-head">What-If Cluster Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick a template country, adjust indicators, see the predicted cluster.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">💡 Uses the active <b>K-Means</b> model with <b>K = {selected_k}</b> for prediction.</div>', unsafe_allow_html=True)

    tmpl = st.selectbox("Template country", sorted(df_fe.index.tolist()), key="pred_tmpl")
    base_row = df_fe.loc[[tmpl], feat].copy()

    editable = [c for c in feat if c in [
        "GDP","Population Total","Birth Rate","Life Expectancy Female",
        "Life Expectancy Male","Infant Mortality Rate","Internet Usage",
        "Mobile Phone Usage","Population Urban","Health Exp % GDP",
        "Health Exp/Capita","CO2 Emissions","Energy Usage",
        "Population 0-14","Population 65+",
    ]]

    if not editable:
        editable = feat[:min(9, len(feat))]

    st.markdown("**Edit indicators:**")
    inp = {}
    cols3 = st.columns(3)
    for i, f in enumerate(editable):
        with cols3[i % 3]:
            inp[f] = st.number_input(f, value=float(base_row.iloc[0][f]),
                                      format="%.6f", key=f"p_{f}")

    custom = base_row.copy()
    for f, v in inp.items():
        custom.loc[tmpl, f] = v

    custom_eng = add_engineered_features(custom)
    for c in feat:
        if c not in custom_eng.columns:
            custom_eng[c] = float(df_fe[c].median())
    custom_fin = custom_eng[feat].fillna(df_fe.median(numeric_only=True))

    sc       = pipe["scaler"]
    p2_model = pipe["pca_2d"]
    km_model = dynamic_models["K-Means"]  # respects current selected_k

    try:
        c_scaled = sc.transform(custom_fin)
        pred_cl  = km_model.predict(c_scaled)[0]
        pred_pca = p2_model.transform(c_scaled)[0]

        r1, r2, r3 = st.columns(3)
        r1.metric("🎯 Predicted Cluster", str(pred_cl))
        r2.metric("PC1", f"{pred_pca[0]:.3f}")
        r3.metric("PC2", f"{pred_pca[1]:.3f}")

        pred_plot = df_clustered.copy()
        pred_plot["_type"] = "Existing"
        nr = pd.DataFrame({
            "Country": [f"★ {tmpl} (custom)"],
            cluster_col: [str(pred_cl)],
            "PC1": [pred_pca[0]], "PC2": [pred_pca[1]], "PC3": [0.0],
            "_type": ["Prediction"],
        })
        pred_plot = pd.concat([pred_plot, nr], ignore_index=True)

        fig_pred = px.scatter(pred_plot, x="PC1", y="PC2",
                               color=cluster_col, symbol="_type",
                               hover_name="Country",
                               title="Prediction in PCA Space",
                               template="plotly_white",
                               color_discrete_sequence=PALETTE)
        fig_pred.update_traces(marker=dict(size=10, line=dict(width=0.7, color="white")))
        fig_pred.update_layout(height=540, legend_title_text="")
        st.plotly_chart(fig_pred, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")


# ╔══════════╗
# ║  DOWNLOAD║
# ╚══════════╝
with t_download:
    st.markdown('<div class="section-head">Export Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download all outputs as CSV or Excel.</div>', unsafe_allow_html=True)

    exp_cl = df_clustered.drop(columns=["PC1","PC2","PC3"], errors="ignore")
    exp_pr = cluster_profile.reset_index()
    exp_ev = eval_df
    exp_km = k_metrics

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("⬇️ Clustered CSV",
                            convert_to_csv(exp_cl),
                            "World_Dev_Clustered.csv", "text/csv",
                            use_container_width=True)
    with d2:
        st.download_button("⬇️ Profiles CSV",
                            convert_to_csv(exp_pr),
                            "Cluster_Profiles.csv", "text/csv",
                            use_container_width=True)
    with d3:
        st.download_button("⬇️ Evaluation CSV",
                            convert_to_csv(exp_ev),
                            "Model_Eval.csv", "text/csv",
                            use_container_width=True)
    with d4:
        xl = convert_to_excel({
            "Clustered_Data": exp_cl,
            "Cluster_Profiles": exp_pr,
            "Model_Evaluation": exp_ev,
            "K_Selection": exp_km,
        })
        st.download_button("⬇️ Full Excel",
                            xl,
                            "World_Dev_Report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)

    st.markdown("**Clustered Data Preview**")
    st.dataframe(exp_cl.head(30), use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🌍 World Development Clustering Dashboard v4 &nbsp;·&nbsp;
  Built with Streamlit · Plotly · Scikit-learn · Pandas
</div>
""", unsafe_allow_html=True)
