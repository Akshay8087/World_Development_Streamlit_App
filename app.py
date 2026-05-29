
# ============================================================
# 🌍 World Development Clustering — Streamlit App v2
# No upload needed — loads clustering_pipeline.pkl automatically
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
# ── Page config ──────────────────────────────────────────────────────────────
 
st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── CSS ──────────────────────────────────────────────────────────────────────
 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
 
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
  .hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 50%, #0a3d5c 100%);
    padding: 36px 40px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute; top: -60px; right: -80px;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(0,180,255,0.18) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 40px; font-weight: 800;
    letter-spacing: -0.5px; margin-bottom: 10px;
  }
  .hero-sub { font-size: 15px; opacity: 0.82; line-height: 1.6; max-width: 900px; }
  .hero-badge {
    display:inline-block; background:rgba(0,180,255,0.2);
    border: 1px solid rgba(0,180,255,0.4);
    color: #7dd3fc; padding: 3px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin-right: 6px; margin-bottom: 14px;
  }
 
  .kpi-card {
    background: white;
    border: 1px solid #e8ecf4;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
  }
  .kpi-label { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-value { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 700; color: #0f172a; }
  .kpi-sub { font-size: 12px; color: #94a3b8; margin-top: 2px; }
 
  .section-head {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 800; color: #0f172a;
    border-left: 4px solid #0ea5e9;
    padding-left: 12px; margin: 20px 0 6px;
  }
  .section-sub { color: #64748b; font-size: 14px; margin-bottom: 18px; }
 
  .info-box {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 12px; padding: 14px 18px;
    color: #0369a1; font-size: 14px; margin-bottom: 16px;
  }
  .warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 12px; padding: 14px 18px;
    color: #92400e; font-size: 14px; margin-bottom: 16px;
  }
 
  .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #e2e8f0; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #64748b !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: none !important;
  }
  .stTabs [aria-selected="true"] {
    background: #0ea5e9 !important;
    color: white !important;
    border-color: #0ea5e9 !important;
  }
  div[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700; }
  div[data-testid="stMetricLabel"] { font-size: 13px !important; }
 
  .cluster-pill {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 600; margin: 3px;
  }
  .footer { text-align:center; color:#94a3b8; font-size:13px; padding:20px 0; border-top:1px solid #e2e8f0; margin-top:30px; }
</style>
""", unsafe_allow_html=True)
 
# ── Constants ─────────────────────────────────────────────────────────────────
 
CURRENCY_COLS = ["GDP", "Health Exp/Capita", "Tourism Inbound", "Tourism Outbound", "Business Tax Rate"]
 
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
 
CLUSTER_NAMES = {
    "0": "🔵 Cluster 0", "1": "🟢 Cluster 1",
    "2": "🟠 Cluster 2", "3": "🔴 Cluster 3",
    "4": "🟣 Cluster 4", "5": "⚫ Cluster 5", "-1": "⬛ Noise",
}
 
PALETTE = px.colors.qualitative.Bold
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def clean_currency_col(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )
 
 
def safe_evaluate(name: str, labels: np.ndarray, X: np.ndarray) -> dict:
    labels = np.asarray(labels)
    mask = labels != -1
    X_valid, lv = X[mask], labels[mask]
    nc = len(set(lv))
    nn = int((labels == -1).sum())
    if nc < 2 or len(X_valid) <= nc:
        return {"Model": name, "Clusters": nc, "Noise": nn,
                "Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    return {
        "Model": name, "Clusters": nc, "Noise": nn,
        "Silhouette": round(silhouette_score(X_valid, lv), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_valid, lv), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(X_valid, lv), 2),
    }
 
 
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GDP" in df.columns and "Population Total" in df.columns:
        df["GDP_per_Capita"] = df["GDP"] / (df["Population Total"] + 1)
    if "Life Expectancy Female" in df.columns and "Life Expectancy Male" in df.columns:
        df["Life_Exp_Gap"] = df["Life Expectancy Female"] - df["Life Expectancy Male"]
        df["Life_Exp_Avg"] = (df["Life Expectancy Female"] + df["Life Expectancy Male"]) / 2
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
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        for name, data in sheets.items():
            data.to_excel(w, sheet_name=name[:31], index=False)
    return out.getvalue()
 
 
# ── Load pipeline ─────────────────────────────────────────────────────────────
 
@st.cache_resource(show_spinner="Loading clustering pipeline...")
def load_pipeline(path: str = "clustering_pipeline.pkl") -> dict:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
 
 
@st.cache_data(show_spinner=False)
def get_k_metrics(_X_scaled: np.ndarray, max_k: int = 10) -> pd.DataFrame:
    max_k = min(max_k, len(_X_scaled) - 1)
    rows = []
    for k in range(2, max_k + 1):
        m = KMeans(n_clusters=k, random_state=42, n_init=15).fit(_X_scaled)
        rows.append({
            "K": k,
            "Inertia": m.inertia_,
            "Silhouette": silhouette_score(_X_scaled, m.labels_),
            "Davies-Bouldin": davies_bouldin_score(_X_scaled, m.labels_),
            "Calinski-Harabasz": calinski_harabasz_score(_X_scaled, m.labels_),
        })
    return pd.DataFrame(rows)
 
 
@st.cache_data(show_spinner=False)
def run_clustering(_X_scaled: np.ndarray, selected_k: int) -> dict:
    km = KMeans(n_clusters=selected_k, random_state=42, n_init=15)
    km_labels = km.fit_predict(_X_scaled)
 
    agg = AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
    agg_labels = agg.fit_predict(_X_scaled)
 
    nbrs = NearestNeighbors(n_neighbors=min(5, len(_X_scaled) - 1)).fit(_X_scaled)
    distances, _ = nbrs.kneighbors(_X_scaled)
    eps_val = float(np.percentile(np.sort(distances[:, -1]), 10))
    dbscan = DBSCAN(eps=eps_val, min_samples=3)
    db_labels = dbscan.fit_predict(_X_scaled)
 
    gmm = GaussianMixture(n_components=selected_k, random_state=42, n_init=5)
    gmm.fit(_X_scaled)
    gmm_labels = gmm.predict(_X_scaled)
 
    eval_df = pd.DataFrame([
        safe_evaluate("K-Means", km_labels, _X_scaled),
        safe_evaluate("Agglomerative", agg_labels, _X_scaled),
        safe_evaluate("DBSCAN", db_labels, _X_scaled),
        safe_evaluate("GMM", gmm_labels, _X_scaled),
    ])
 
    return {
        "kmeans": km, "agg": agg, "dbscan": dbscan, "gmm": gmm,
        "km_labels": km_labels, "agg_labels": agg_labels,
        "db_labels": db_labels, "gmm_labels": gmm_labels,
        "eps_val": eps_val, "eval_df": eval_df,
    }
 
 
# ── Load pipeline from disk ───────────────────────────────────────────────────
 
pipe = load_pipeline("clustering_pipeline.pkl")
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
 
with st.sidebar:
    st.markdown("## 🌍 Dashboard Controls")
 
    # Optional data upload (override pkl)
    st.markdown("### 📂 Data Source")
    upload_mode = st.radio("Data source", ["Use built-in dataset", "Upload my own CSV/Excel"], index=0)
    uploaded_file = None
    if upload_mode == "Upload my own CSV/Excel":
        uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            st.success("Custom file uploaded ✅")
 
    st.divider()
    st.markdown("### ⚙️ Clustering Settings")
 
    k_mode = st.radio("K selection", ["Auto (Silhouette)", "Manual"], index=0)
    manual_k = st.slider("Manual K", 2, 10, 4, disabled=(k_mode == "Auto (Silhouette)"))
 
    model_choice = st.selectbox(
        "Primary model",
        ["K-Means", "Agglomerative", "GMM", "DBSCAN"], index=0
    )
 
    st.divider()
    st.markdown("### 📌 Features")
    for feat in ["✅ Auto-loaded pre-trained pipeline", "✅ Upload override supported",
                 "✅ 4 clustering algorithms", "✅ PCA 2D + 3D views",
                 "✅ Cluster profiling & radar", "✅ Country deep dive",
                 "✅ What-if predictor", "✅ Excel/CSV export"]:
        st.caption(feat)
 
# ── Resolve data source ───────────────────────────────────────────────────────
 
@st.cache_data(show_spinner=False)
def preprocess_uploaded(raw: pd.DataFrame) -> dict:
    """Full preprocessing for user-uploaded data."""
    df_proc = raw.copy()
    for col in CURRENCY_COLS:
        if col in df_proc.columns:
            df_proc[col] = clean_currency_col(df_proc[col])
 
    numeric_cols = [c for c in df_proc.select_dtypes(include=[np.number]).columns
                    if c not in ("Number of Records", "Year")]
    if len(numeric_cols) < 3:
        raise ValueError("Need at least 3 numeric columns.")
 
    df_country = df_proc.groupby("Country")[numeric_cols].mean()
    miss = df_country.isnull().mean()
    drop_cols = miss[miss > 0.5].index.tolist()
    df_clean = df_country.drop(columns=drop_cols)
 
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df_clean),
                               index=df_clean.index, columns=df_clean.columns)
 
    df_fe = add_engineered_features(df_imputed)
    df_fe = df_fe.fillna(df_fe.median(numeric_only=True))
    feature_cols = df_fe.columns.tolist()
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_fe)
 
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca2 = pca_2d.fit_transform(X_scaled)
    pca_3d = PCA(n_components=3, random_state=42)
    X_pca3 = pca_3d.fit_transform(X_scaled)
 
    return {
        "df_raw": raw, "df_proc": df_proc, "df_country": df_country,
        "df_clean": df_clean, "df_fe": df_fe, "drop_cols": drop_cols,
        "imputer": imputer, "scaler": scaler, "feature_cols": feature_cols,
        "X_scaled": X_scaled, "pca_2d": pca_2d, "pca_3d": pca_3d,
        "X_pca2": X_pca2, "X_pca3": X_pca3,
    }
 
 
# Determine which data to use
use_custom = False
if upload_mode == "Upload my own CSV/Excel" and uploaded_file is not None:
    try:
        name = uploaded_file.name.lower()
        raw_uploaded = pd.read_csv(uploaded_file) if name.endswith(".csv") else pd.read_excel(uploaded_file)
        if "Country" not in raw_uploaded.columns:
            st.error("❌ Uploaded file must have a 'Country' column.")
            st.stop()
        with st.spinner("Processing uploaded file..."):
            prep = preprocess_uploaded(raw_uploaded)
        use_custom = True
    except Exception as e:
        st.error(f"Error processing upload: {e}")
        st.stop()
elif pipe is not None:
    prep = pipe
else:
    st.error("❌ `clustering_pipeline.pkl` not found. Please place it next to `app.py`.")
    st.stop()
 
# ── Extract prep objects ──────────────────────────────────────────────────────
 
df_raw      = prep["df_raw"]
df_fe       = prep["df_fe"]
X_scaled    = prep["X_scaled"]
X_pca2      = prep["X_pca2"]
X_pca3      = prep["X_pca3"]
pca_2d      = prep["pca_2d"]
pca_3d      = prep["pca_3d"]
scaler      = prep["scaler"]
feature_cols = prep["feature_cols"]
 
# ── K selection & clustering ──────────────────────────────────────────────────
 
with st.spinner("Running models..."):
    k_metrics = get_k_metrics(X_scaled, max_k=min(10, len(df_fe) - 1))
    auto_k = int(k_metrics.loc[k_metrics["Silhouette"].idxmax(), "K"])
    selected_k = auto_k if k_mode == "Auto (Silhouette)" else manual_k
    selected_k = max(2, min(selected_k, len(df_fe) - 1))
    cluster_results = run_clustering(X_scaled, selected_k)
 
label_map = {
    "K-Means": cluster_results["km_labels"],
    "Agglomerative": cluster_results["agg_labels"],
    "GMM": cluster_results["gmm_labels"],
    "DBSCAN": cluster_results["db_labels"],
}
 
cluster_col = f"{model_choice}_Cluster"
primary_labels = label_map[model_choice]
 
# Build clustered df
df_clustered = df_fe.copy()
df_clustered[cluster_col] = primary_labels
df_clustered["PC1"] = X_pca2[:, 0]
df_clustered["PC2"] = X_pca2[:, 1]
df_clustered["PC3"] = X_pca3[:, 2] if X_pca3.shape[1] > 2 else 0
df_clustered = df_clustered.reset_index().rename(columns={"index": "Country"})
df_clustered[cluster_col] = df_clustered[cluster_col].astype(str)
 
profile_src = df_fe.copy()
profile_src[cluster_col] = primary_labels
cluster_profile = profile_src.groupby(cluster_col)[feature_cols].mean()
 
n_clusters = len(set(primary_labels)) - (1 if -1 in set(primary_labels) else 0)
n_noise = int((np.asarray(primary_labels) == -1).sum())
pca_var = pca_2d.explained_variance_ratio_.sum() * 100
 
# ── Hero banner ───────────────────────────────────────────────────────────────
 
data_label = "Custom Upload" if use_custom else "Built-in Dataset"
st.markdown(f"""
<div class="hero">
  <span class="hero-badge">🌍 Unsupervised ML</span>
  <span class="hero-badge">📊 {selected_k} Clusters</span>
  <span class="hero-badge">🗃️ {data_label}</span>
  <div class="hero-title">World Development Clustering</div>
  <div class="hero-sub">
    Grouping {len(df_fe):,} countries by economic, health, digital &amp; demographic indicators.
    Pipeline: clean → engineer → scale → PCA → cluster → profile.
    Switch algorithms and K from the sidebar — no upload required.
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── KPI row ───────────────────────────────────────────────────────────────────
 
k1, k2, k3, k4, k5, k6 = st.columns(6)
metrics = [
    ("Countries", f"{len(df_fe):,}", "After aggregation"),
    ("Features", f"{len(feature_cols)}", "Post engineering"),
    ("Clusters (K)", str(selected_k), k_mode),
    ("Best Auto K", str(auto_k), "By silhouette"),
    ("PCA Variance", f"{pca_var:.1f}%", "2 components"),
    ("Noise Points", str(n_noise), "DBSCAN only"),
]
for col, (label, val, sub) in zip([k1, k2, k3, k4, k5, k6], metrics):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
if n_noise > 0:
    st.markdown(f'<div class="warn-box">⚠️ DBSCAN identified <b>{n_noise}</b> noise/outlier countries (label = -1). These appear only in DBSCAN view.</div>', unsafe_allow_html=True)
 
# ── Tabs ──────────────────────────────────────────────────────────────────────
 
(tab_overview, tab_eda, tab_cluster, tab_profile,
 tab_country, tab_predict, tab_download) = st.tabs([
    "📌 Overview", "🔍 EDA", "🤖 Clustering",
    "📊 Profiles", "🗺️ Country Explorer", "🔮 Predict", "⬇️ Download",
])
 
# ╔══════════════════════════════════════╗
# ║  TAB 1 — OVERVIEW                   ║
# ╚══════════════════════════════════════╝
with tab_overview:
    st.markdown('<div class="section-head">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Business context, ML workflow, and pipeline summary.</div>', unsafe_allow_html=True)
 
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("""
**🎯 Goal** — Group countries into meaningful development tiers using unsupervised ML.
 
**🧠 Why clustering?** — No predefined labels like *developed* or *low-income* exist.
Clustering lets the data reveal natural groupings.
 
**🔁 Pipeline steps**
""")
        steps = pd.DataFrame({
            "Step": ["Data Loading", "Currency Cleaning", "Country Aggregation",
                     "Missing-value Imputation", "Feature Engineering", "StandardScaler",
                     "PCA (2D / 3D)", "Clustering × 4", "Evaluation", "Profiling"],
            "Detail": [
                "CSV / Excel / Pre-built PKL",
                "Strip $, %, commas → numeric",
                "Mean per country across years",
                "Median strategy, drop >50% missing cols",
                "GDP/capita, digital access, life-exp gap, etc.",
                "Zero mean, unit variance",
                "Reduce to 2–3 components for visualisation",
                "K-Means · Agglomerative · DBSCAN · GMM",
                "Silhouette · Davies-Bouldin · Calinski-Harabasz",
                "Per-cluster mean features + radar chart",
            ],
        })
        st.dataframe(steps, use_container_width=True, hide_index=True)
 
        if prep.get("drop_cols"):
            st.markdown(f"**Columns dropped (>50% missing):** `{'`, `'.join(prep['drop_cols'])}`")
        else:
            st.markdown('<div class="info-box">✅ No columns dropped — all features passed the missing-value threshold.</div>', unsafe_allow_html=True)
 
    with c2:
        st.markdown("**Raw data preview (first 10 rows)**")
        st.dataframe(df_raw.head(10), use_container_width=True)
 
 
# ╔══════════════════════════════════════╗
# ║  TAB 2 — EDA                        ║
# ╚══════════════════════════════════════╝
with tab_eda:
    st.markdown('<div class="section-head">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Missing values, feature distributions, and correlations.</div>', unsafe_allow_html=True)
 
    e1, e2 = st.columns([1, 1])
    with e1:
        st.markdown("**Missing Values**")
        miss_pct = (df_raw.isnull().mean() * 100).sort_values(ascending=False)
        miss_pct = miss_pct[miss_pct > 0]
        if miss_pct.empty:
            st.success("No missing values in the dataset.")
        else:
            fig_m = px.bar(
                miss_pct.reset_index(), x=0, y="index", orientation="h",
                labels={"index": "Feature", 0: "Missing %"},
                title="Missing Value % by Feature", template="plotly_white",
                color=0, color_continuous_scale="Blues",
            )
            fig_m.update_layout(height=max(400, 22 * len(miss_pct)), yaxis=dict(autorange="reversed"),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_m, use_container_width=True)
 
    with e2:
        st.markdown("**Feature Distribution**")
        num_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns
                    if c not in ("Number of Records", "Year")]
        sel_feat = st.selectbox("Select feature", num_cols, key="eda_feat")
        fig_h = px.histogram(df_raw, x=sel_feat, nbins=35, marginal="box",
                              title=f"{sel_feat} — Distribution",
                              template="plotly_white", color_discrete_sequence=["#0ea5e9"])
        fig_h.update_layout(height=460, title_x=0.02)
        st.plotly_chart(fig_h, use_container_width=True)
 
    st.markdown("**Feature Correlation Heatmap**")
    with st.expander("Show heatmap", expanded=False):
        corr = df_fe.corr(numeric_only=True)
        fig_c = px.imshow(corr, text_auto=".2f", aspect="auto",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                           title="Correlation Heatmap — Engineered Features")
        fig_c.update_layout(height=800, template="plotly_white")
        st.plotly_chart(fig_c, use_container_width=True)
 
    st.markdown("**Pairplot (top 4 features)**")
    with st.expander("Show pairplot", expanded=False):
        pp_feats = [f for f in ["GDP_per_Capita", "Life_Exp_Avg", "Infant Mortality Rate",
                                 "Internet Usage", "Birth Rate"] if f in df_clustered.columns][:4]
        if len(pp_feats) >= 2:
            fig_pp = px.scatter_matrix(df_clustered, dimensions=pp_feats,
                                        color=cluster_col, template="plotly_white",
                                        color_discrete_sequence=PALETTE)
            fig_pp.update_layout(height=700)
            st.plotly_chart(fig_pp, use_container_width=True)
 
 
# ╔══════════════════════════════════════╗
# ║  TAB 3 — CLUSTERING                 ║
# ╚══════════════════════════════════════╝
with tab_cluster:
    st.markdown('<div class="section-head">Clustering Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">PCA projections, model evaluation, and K selection.</div>', unsafe_allow_html=True)
 
    vis1, vis2 = st.columns([1.2, 0.8])
 
    with vis1:
        fig_2d = px.scatter(
            df_clustered, x="PC1", y="PC2", color=cluster_col,
            hover_name="Country", hover_data={cluster_col: True, "PC1": ":.2f", "PC2": ":.2f"},
            title=f"{model_choice} — PCA 2D Scatter",
            template="plotly_white", color_discrete_sequence=PALETTE,
        )
        fig_2d.update_traces(marker=dict(size=9, line=dict(width=0.7, color="white")))
        fig_2d.update_layout(height=560, title_x=0.02, legend_title_text="Cluster")
        st.plotly_chart(fig_2d, use_container_width=True)
 
    with vis2:
        st.markdown("**Model Evaluation**")
        eval_df = cluster_results["eval_df"].copy()
 
        def style_eval(df):
            styled = df.style
            if "Silhouette" in df.columns:
                styled = styled.background_gradient(subset=["Silhouette"], cmap="Greens")
            if "Davies-Bouldin" in df.columns:
                styled = styled.background_gradient(subset=["Davies-Bouldin"], cmap="RdYlGn_r")
            return styled
 
        st.dataframe(style_eval(eval_df), use_container_width=True, hide_index=True)
 
        fig_ev = px.bar(
            eval_df.melt(id_vars="Model", value_vars=["Silhouette", "Davies-Bouldin"]),
            x="Model", y="value", color="variable", barmode="group",
            title="Silhouette vs Davies-Bouldin", template="plotly_white",
            color_discrete_sequence=["#0ea5e9", "#f97316"],
        )
        fig_ev.update_layout(height=360, title_x=0.02, legend_title_text="")
        st.plotly_chart(fig_ev, use_container_width=True)
 
    # K selection charts
    st.markdown('<div class="section-head">K Selection Analysis</div>', unsafe_allow_html=True)
    k1c, k2c = st.columns(2)
    with k1c:
        fig_sil = px.line(k_metrics, x="K", y="Silhouette", markers=True,
                           title=f"Silhouette Score — Best K = {auto_k}",
                           template="plotly_white", color_discrete_sequence=["#0ea5e9"])
        fig_sil.add_vline(x=auto_k, line_dash="dash", line_color="#22c55e",
                           annotation_text=f"Auto K={auto_k}", annotation_position="top right")
        fig_sil.update_layout(height=380, title_x=0.02)
        st.plotly_chart(fig_sil, use_container_width=True)
    with k2c:
        fig_el = px.line(k_metrics, x="K", y="Inertia", markers=True,
                          title="Elbow — Inertia by K", template="plotly_white",
                          color_discrete_sequence=["#f97316"])
        fig_el.update_layout(height=380, title_x=0.02)
        st.plotly_chart(fig_el, use_container_width=True)
 
    with st.expander("🧊 3D PCA View", expanded=False):
        fig_3d = px.scatter_3d(
            df_clustered, x="PC1", y="PC2", z="PC3", color=cluster_col,
            hover_name="Country", title=f"{model_choice} — PCA 3D",
            template="plotly_white", color_discrete_sequence=PALETTE,
        )
        fig_3d.update_traces(marker=dict(size=5))
        fig_3d.update_layout(height=650)
        st.plotly_chart(fig_3d, use_container_width=True)
 
 
# ╔══════════════════════════════════════╗
# ║  TAB 4 — PROFILES                   ║
# ╚══════════════════════════════════════╝
with tab_profile:
    st.markdown('<div class="section-head">Cluster Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average feature values, heatmap, and radar chart per cluster.</div>', unsafe_allow_html=True)
 
    p1, p2 = st.columns([1, 1])
    with p1:
        size_df = (df_clustered.groupby(cluster_col)["Country"]
                   .count().reset_index(name="Countries").sort_values(cluster_col))
        fig_sz = px.bar(size_df, x=cluster_col, y="Countries", text="Countries",
                         title="Cluster Sizes", template="plotly_white",
                         color=cluster_col, color_discrete_sequence=PALETTE)
        fig_sz.update_traces(textposition="outside")
        fig_sz.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_sz, use_container_width=True)
 
    with p2:
        st.markdown("**Cluster Interpretation Guide**")
        st.markdown("""
| Signal | Likely cluster type |
|---|---|
| High life expectancy, high internet, high GDP/capita | 🔵 Highly Developed |
| Mid-range across all indicators | 🟢 Upper-Middle |
| Moderate growth, urbanising | 🟠 Developing |
| High birth rate, high infant mortality, low digital | 🔴 Low Development |
""")
        st.markdown("**Mean feature values per cluster**")
        st.dataframe(cluster_profile.round(2), use_container_width=True)
 
    # Profile heatmap
    st.markdown('<div class="section-head">Profile Heatmap (Z-scored)</div>', unsafe_allow_html=True)
    feats_h = [f for f in KEY_PROFILE_FEATURES if f in cluster_profile.columns]
    if not feats_h:
        feats_h = cluster_profile.columns[:10].tolist()
    prof_z = (cluster_profile[feats_h] - cluster_profile[feats_h].mean()) / (cluster_profile[feats_h].std() + 1e-9)
    fig_heat = px.imshow(prof_z.T, text_auto=".2f", aspect="auto",
                          color_continuous_scale="RdYlGn", zmin=-2, zmax=2,
                          labels=dict(x="Cluster", y="Feature", color="Z-score"),
                          title="Relative Strengths per Cluster")
    fig_heat.update_layout(height=540, template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)
 
    # Radar
    feats_r = [f for f in RADAR_FEATURES if f in cluster_profile.columns]
    if len(feats_r) >= 3:
        radar = cluster_profile[feats_r].copy()
        for col in radar.columns:
            radar[col] = (radar[col] - radar[col].min()) / (radar[col].max() - radar[col].min() + 1e-9)
        fig_r = go.Figure()
        for cid, row in radar.iterrows():
            vals = row.tolist()
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=feats_r + [feats_r[0]],
                fill="toself", name=f"Cluster {cid}",
            ))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                              height=580, template="plotly_white",
                              title="Radar — Normalised Cluster Comparison")
        st.plotly_chart(fig_r, use_container_width=True)
 
 
# ╔══════════════════════════════════════╗
# ║  TAB 5 — COUNTRY EXPLORER           ║
# ╚══════════════════════════════════════╝
with tab_country:
    st.markdown('<div class="section-head">Country Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search, filter, and deep-dive into any country.</div>', unsafe_allow_html=True)
 
    f1, f2, f3 = st.columns([1.3, 1, 0.8])
    with f1:
        search = st.text_input("🔍 Search country name", "")
    with f2:
        cluster_opts = ["All"] + sorted(df_clustered[cluster_col].unique().tolist())
        clust_filter = st.selectbox("Filter by cluster", cluster_opts)
    with f3:
        top_n = st.slider("Rows to show", 10, min(300, len(df_clustered)), 50)
 
    filtered = df_clustered.copy()
    if search.strip():
        filtered = filtered[filtered["Country"].str.contains(search.strip(), case=False, na=False)]
    if clust_filter != "All":
        filtered = filtered[filtered[cluster_col] == clust_filter]
 
    disp_cols = ["Country", cluster_col, "PC1", "PC2"]
    imp = ["GDP_per_Capita", "Life_Exp_Avg", "Internet Usage", "Infant Mortality Rate",
           "Birth Rate", "Population Urban", "Digital_Access"]
    disp_cols += [c for c in imp if c in filtered.columns]
 
    st.dataframe(filtered[disp_cols].head(top_n), use_container_width=True, hide_index=True)
 
    # Deep dive
    st.markdown('<div class="section-head">Single Country Deep Dive</div>', unsafe_allow_html=True)
    sel_country = st.selectbox("Select country", df_clustered["Country"].sort_values().tolist(), key="deep_dive")
    crow = df_clustered[df_clustered["Country"] == sel_country].iloc[0]
 
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Cluster", crow[cluster_col])
    d2.metric("PC1", f"{crow['PC1']:.2f}")
    d3.metric("PC2", f"{crow['PC2']:.2f}")
    d4.metric("GDP/Capita", f"{crow['GDP_per_Capita']:,.0f}" if "GDP_per_Capita" in crow else "—")
    d5.metric("Internet Usage", f"{crow['Internet Usage']:.1f}%" if "Internet Usage" in crow else "—")
 
    sel_feats = [c for c in imp if c in df_clustered.columns]
    if sel_feats:
        cvals = pd.DataFrame({"Feature": sel_feats, "Value": [crow[c] for c in sel_feats]})
        fig_cv = px.bar(cvals, x="Feature", y="Value",
                         title=f"Key Indicators — {sel_country}",
                         template="plotly_white", color="Value",
                         color_continuous_scale="Blues")
        fig_cv.update_layout(height=420, xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)
 
    # Geo map
    st.markdown("**Global Cluster Map**")
    map_df = df_clustered[["Country", cluster_col]].copy()
    fig_geo = px.choropleth(
        map_df, locations="Country", locationmode="country names",
        color=cluster_col, title="Cluster by Country — World Map",
        template="plotly_white", color_discrete_sequence=PALETTE,
    )
    fig_geo.update_layout(height=500)
    st.plotly_chart(fig_geo, use_container_width=True)
 
 
# ╔══════════════════════════════════════╗
# ║  TAB 6 — PREDICT                    ║
# ╚══════════════════════════════════════╝
with tab_predict:
    st.markdown('<div class="section-head">What-If Cluster Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick a country as a template, tweak indicators, and see which cluster it lands in.</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Prediction uses <b>K-Means</b> (most interpretable for deployment). Change any input and the cluster updates live.</div>', unsafe_allow_html=True)
 
    tmpl = st.selectbox("Template country", df_fe.index.sort_values().tolist(), key="pred_tmpl")
    base = df_fe.loc[[tmpl], feature_cols].copy()
 
    editable = [c for c in [
        "GDP", "Population Total", "Birth Rate",
        "Life Expectancy Female", "Life Expectancy Male", "Infant Mortality Rate",
        "Internet Usage", "Mobile Phone Usage", "Population Urban",
        "Health Exp % GDP", "Health Exp/Capita", "CO2 Emissions",
        "Energy Usage", "Population 0-14", "Population 65+",
    ] if c in base.columns]
 
    st.markdown("**Edit indicators:**")
    inp = {}
    cols3 = st.columns(3)
    for i, feat in enumerate(editable):
        with cols3[i % 3]:
            inp[feat] = st.number_input(feat, value=float(base.iloc[0][feat]),
                                         format="%.4f", key=f"pred_{feat}")
 
    custom = base.copy()
    for f, v in inp.items():
        custom.loc[tmpl, f] = v
 
    custom_eng = add_engineered_features(custom)
    for c in feature_cols:
        if c not in custom_eng.columns:
            custom_eng[c] = df_fe[c].median()
    custom_eng = custom_eng[feature_cols].fillna(df_fe.median(numeric_only=True))
 
    cscaled = scaler.transform(custom_eng)
    pred_cl = cluster_results["kmeans"].predict(cscaled)[0]
    pred_pca = pca_2d.transform(cscaled)[0]
 
    r1, r2, r3 = st.columns(3)
    r1.metric("🎯 Predicted Cluster", str(pred_cl))
    r2.metric("PC1", f"{pred_pca[0]:.2f}")
    r3.metric("PC2", f"{pred_pca[1]:.2f}")
 
    pred_plot = df_clustered.copy()
    pred_plot["_type"] = "Existing"
    new_row = pd.DataFrame({
        "Country": [f"★ {tmpl} (custom)"],
        cluster_col: [str(pred_cl)],
        "PC1": [pred_pca[0]], "PC2": [pred_pca[1]], "PC3": [0],
        "_type": ["Prediction"],
    })
    pred_plot = pd.concat([pred_plot, new_row], ignore_index=True)
 
    fig_pred = px.scatter(
        pred_plot, x="PC1", y="PC2", color=cluster_col, symbol="_type",
        hover_name="Country", title="Prediction in PCA Space",
        template="plotly_white", color_discrete_sequence=PALETTE,
    )
    fig_pred.update_traces(marker=dict(size=10, line=dict(width=0.8, color="white")))
    fig_pred.update_layout(height=560, title_x=0.02, legend_title_text="")
    st.plotly_chart(fig_pred, use_container_width=True)
 
 
# ╔══════════════════════════════════════╗
# ║  TAB 7 — DOWNLOAD                   ║
# ╚══════════════════════════════════════╝
with tab_download:
    st.markdown('<div class="section-head">Export Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download clustered data, profiles, and model evaluation.</div>', unsafe_allow_html=True)
 
    exp_clustered  = df_clustered.drop(columns=["PC1", "PC2", "PC3"], errors="ignore")
    exp_profiles   = cluster_profile.reset_index()
    exp_eval       = cluster_results["eval_df"]
    exp_k_metrics  = k_metrics
 
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("⬇️ Clustered CSV", convert_to_csv(exp_clustered),
                            "World_Dev_Clustered.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇️ Profiles CSV", convert_to_csv(exp_profiles),
                            "Cluster_Profiles.csv", "text/csv", use_container_width=True)
    with d3:
        st.download_button("⬇️ Evaluation CSV", convert_to_csv(exp_eval),
                            "Model_Evaluation.csv", "text/csv", use_container_width=True)
    with d4:
        xl = convert_to_excel({
            "Clustered_Data": exp_clustered,
            "Cluster_Profiles": exp_profiles,
            "Model_Evaluation": exp_eval,
            "K_Selection": exp_k_metrics,
        })
        st.download_button("⬇️ Full Excel Report", xl,
                            "World_Dev_Clustering_Report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)
 
    st.markdown("**Preview — Clustered Data**")
    st.dataframe(exp_clustered.head(30), use_container_width=True, hide_index=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
 
st.markdown("""
<div class="footer">
  🌍 World Development Clustering Dashboard v2 &nbsp;·&nbsp;
  Built with Streamlit · Plotly · Scikit-learn · Pandas
</div>
""", unsafe_allow_html=True)
