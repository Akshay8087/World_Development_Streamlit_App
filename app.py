# ============================================================
# 🌍 World Development Clustering — Professional Streamlit App
# Author: Akshay Project
# Purpose: Cluster countries using development and economic indicators
# ============================================================

import io
import warnings
from typing import Dict, List, Tuple

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

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="World Development Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {background-color: #f7f9fc;}
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
            padding: 30px 32px;
            border-radius: 24px;
            color: white;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.22);
            margin-bottom: 20px;
        }
        .hero-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.8px;
        }
        .hero-subtitle {
            font-size: 17px;
            opacity: 0.92;
            line-height: 1.55;
            max-width: 1050px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 18px;
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.07);
            border: 1px solid #e5e7eb;
        }
        .section-title {
            font-size: 25px;
            font-weight: 800;
            color: #0f172a;
            margin-top: 8px;
            margin-bottom: 4px;
        }
        .section-caption {
            color: #475569;
            font-size: 15px;
            margin-bottom: 18px;
        }
        div[data-testid="stMetricValue"] {font-size: 28px;}
        div[data-testid="stMetricLabel"] {font-size: 14px; color: #475569;}
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 14px 14px 0 0;
            padding: 12px 18px;
            border: 1px solid #e5e7eb;
        }
        .stTabs [aria-selected="true"] {
            background: #e0f2fe;
            color: #075985;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Helper Functions
# ============================================================

CURRENCY_COLS = [
    "GDP",
    "Health Exp/Capita",
    "Tourism Inbound",
    "Tourism Outbound",
    "Business Tax Rate",
]

KEY_PROFILE_FEATURES = [
    "Birth Rate",
    "CO2 Emissions",
    "GDP",
    "Health Exp % GDP",
    "Infant Mortality Rate",
    "Internet Usage",
    "Life Expectancy Female",
    "Population Urban",
    "GDP_per_Capita",
    "Digital_Access",
    "Youthfulness_Index",
]

RADAR_FEATURES = [
    "Birth Rate",
    "Life Expectancy Female",
    "Infant Mortality Rate",
    "Internet Usage",
    "Population Urban",
    "GDP_per_Capita",
    "Health Exp % GDP",
    "Digital_Access",
]


def clean_currency_col(series: pd.Series) -> pd.Series:
    """Convert string currency/percent columns into numeric values."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"[$,%]", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel data from Streamlit uploader."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel file.")


def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "Dataset is empty."
    if "Country" not in df.columns:
        return False, "Country column is required. Please upload the original project dataset."
    numeric_count = df.select_dtypes(include=[np.number]).shape[1]
    if numeric_count < 3 and not any(col in df.columns for col in CURRENCY_COLS):
        return False, "Dataset needs development indicator columns for clustering."
    return True, "Data validated successfully."


@st.cache_data(show_spinner=False)
def preprocess_data(df_raw: pd.DataFrame) -> Dict[str, object]:
    """
    Clean raw data, aggregate country-level data, create engineered features,
    scale features, and compute PCA representations.
    """
    df_proc = df_raw.copy()

    # Clean currency / percentage-like columns
    for col in CURRENCY_COLS:
        if col in df_proc.columns:
            df_proc[col] = clean_currency_col(df_proc[col])

    # Keep numeric columns only for clustering
    numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Number of Records"]

    if len(numeric_cols) < 3:
        raise ValueError("Not enough numeric columns after cleaning. Need at least 3 numeric features.")

    # Aggregate multiple years/records to country level
    df_country = df_proc.groupby("Country")[numeric_cols].mean()

    # Drop columns with >50% missing values
    missing_frac = df_country.isnull().mean()
    drop_cols = missing_frac[missing_frac > 0.50].index.tolist()
    df_clean = df_country.drop(columns=drop_cols)

    # Median imputation
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_clean),
        index=df_clean.index,
        columns=df_clean.columns,
    )

    # Feature engineering
    df_fe = df_imputed.copy()

    if "GDP" in df_fe.columns and "Population Total" in df_fe.columns:
        df_fe["GDP_per_Capita"] = df_fe["GDP"] / (df_fe["Population Total"] + 1)

    if "Life Expectancy Female" in df_fe.columns and "Life Expectancy Male" in df_fe.columns:
        df_fe["Life_Exp_Gap"] = df_fe["Life Expectancy Female"] - df_fe["Life Expectancy Male"]
        df_fe["Life_Exp_Avg"] = (
            df_fe["Life Expectancy Female"] + df_fe["Life Expectancy Male"]
        ) / 2

    if "Health Exp % GDP" in df_fe.columns and "Health Exp/Capita" in df_fe.columns:
        df_fe["Health_Investment_Index"] = df_fe["Health Exp % GDP"] * np.log1p(
            df_fe["Health Exp/Capita"].clip(lower=0)
        )

    if "Population 0-14" in df_fe.columns and "Population 65+" in df_fe.columns:
        df_fe["Youthfulness_Index"] = df_fe["Population 0-14"] / (
            df_fe["Population 65+"] + 0.001
        )

    if "Internet Usage" in df_fe.columns and "Mobile Phone Usage" in df_fe.columns:
        df_fe["Digital_Access"] = (
            df_fe["Internet Usage"] + df_fe["Mobile Phone Usage"]
        ) / 2

    if "CO2 Emissions" in df_fe.columns and "Energy Usage" in df_fe.columns:
        df_fe["CO2_Intensity"] = df_fe["CO2 Emissions"] / (df_fe["Energy Usage"] + 1)

    # Replace infinite values created by ratios
    df_fe = df_fe.replace([np.inf, -np.inf], np.nan)
    df_fe = df_fe.fillna(df_fe.median(numeric_only=True))

    feature_cols = df_fe.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_fe)

    pca_2d = PCA(n_components=2, random_state=42)
    X_pca2 = pca_2d.fit_transform(X_scaled)

    pca_3d = PCA(n_components=3, random_state=42)
    X_pca3 = pca_3d.fit_transform(X_scaled)

    return {
        "df_proc": df_proc,
        "df_country": df_country,
        "df_clean": df_clean,
        "df_fe": df_fe,
        "drop_cols": drop_cols,
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_scaled": X_scaled,
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "X_pca2": X_pca2,
        "X_pca3": X_pca3,
    }


def safe_evaluate(name: str, labels: np.ndarray, X: np.ndarray) -> Dict[str, object]:
    """Evaluate clustering labels safely, including DBSCAN noise handling."""
    labels = np.asarray(labels)
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    n_clusters = len(set(labels_valid))
    n_noise = int((labels == -1).sum())

    if n_clusters < 2 or len(X_valid) <= n_clusters:
        return {
            "Model": name,
            "Clusters": n_clusters,
            "Noise": n_noise,
            "Silhouette": np.nan,
            "Davies-Bouldin": np.nan,
            "Calinski-Harabasz": np.nan,
        }

    return {
        "Model": name,
        "Clusters": n_clusters,
        "Noise": n_noise,
        "Silhouette": round(silhouette_score(X_valid, labels_valid), 4),
        "Davies-Bouldin": round(davies_bouldin_score(X_valid, labels_valid), 4),
        "Calinski-Harabasz": round(calinski_harabasz_score(X_valid, labels_valid), 2),
    }


@st.cache_data(show_spinner=False)
def find_optimal_k(X_scaled: np.ndarray, min_k: int = 2, max_k: int = 10) -> pd.DataFrame:
    """Calculate K-Means metrics for different K values."""
    max_k = min(max_k, len(X_scaled) - 1)
    results = []

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=15, max_iter=300)
        labels = model.fit_predict(X_scaled)
        results.append(
            {
                "K": k,
                "Inertia": model.inertia_,
                "Silhouette": silhouette_score(X_scaled, labels),
                "Davies-Bouldin": davies_bouldin_score(X_scaled, labels),
                "Calinski-Harabasz": calinski_harabasz_score(X_scaled, labels),
            }
        )

    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def run_clustering(X_scaled: np.ndarray, selected_k: int) -> Dict[str, object]:
    """Run K-Means, Agglomerative, DBSCAN, and GMM."""
    # K-Means
    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=15, max_iter=300)
    km_labels = kmeans.fit_predict(X_scaled)

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)

    # DBSCAN eps from k-distance percentile
    k_neighbors = min(5, len(X_scaled) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_dists = np.sort(distances[:, k_neighbors - 1])
    eps_val = float(np.percentile(k_dists, 10))
    dbscan = DBSCAN(eps=eps_val, min_samples=3)
    db_labels = dbscan.fit_predict(X_scaled)

    # GMM
    gmm = GaussianMixture(
        n_components=selected_k,
        random_state=42,
        covariance_type="full",
        n_init=5,
    )
    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    gmm_proba = gmm.predict_proba(X_scaled)

    eval_df = pd.DataFrame(
        [
            safe_evaluate("K-Means", km_labels, X_scaled),
            safe_evaluate("Agglomerative", agg_labels, X_scaled),
            safe_evaluate("DBSCAN", db_labels, X_scaled),
            safe_evaluate("GMM", gmm_labels, X_scaled),
        ]
    )

    return {
        "kmeans": kmeans,
        "agg": agg,
        "dbscan": dbscan,
        "gmm": gmm,
        "km_labels": km_labels,
        "agg_labels": agg_labels,
        "db_labels": db_labels,
        "gmm_labels": gmm_labels,
        "gmm_proba": gmm_proba,
        "eps_val": eps_val,
        "eval_df": eval_df,
    }


def build_clustered_df(
    df_fe: pd.DataFrame,
    X_pca2: np.ndarray,
    X_pca3: np.ndarray,
    labels: np.ndarray,
    label_name: str,
) -> pd.DataFrame:
    result = df_fe.copy()
    result[label_name] = labels
    result["PC1"] = X_pca2[:, 0]
    result["PC2"] = X_pca2[:, 1]
    result["PC3"] = X_pca3[:, 2]
    return result.reset_index().rename(columns={"index": "Country"})


def plot_pca_2d(df_plot: pd.DataFrame, cluster_col: str, title: str):
    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color=cluster_col,
        hover_name="Country",
        hover_data=[cluster_col],
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.7, color="white")))
    fig.update_layout(height=620, title_x=0.02, legend_title_text="Cluster")
    return fig


def plot_pca_3d(df_plot: pd.DataFrame, cluster_col: str, title: str):
    fig = px.scatter_3d(
        df_plot,
        x="PC1",
        y="PC2",
        z="PC3",
        color=cluster_col,
        hover_name="Country",
        hover_data=[cluster_col],
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0.4, color="white")))
    fig.update_layout(height=650, title_x=0.02, legend_title_text="Cluster")
    return fig


def plot_missing_values(df: pd.DataFrame):
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        return None

    fig = px.bar(
        missing.reset_index(),
        x=0,
        y="index",
        orientation="h",
        labels={"index": "Feature", 0: "Missing %"},
        title="Missing Values by Feature",
        template="plotly_white",
        color=0,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(420, 24 * len(missing)), yaxis=dict(autorange="reversed"))
    return fig


def plot_correlation(df_fe: pd.DataFrame):
    corr = df_fe.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(height=800, template="plotly_white", title_x=0.02)
    return fig


def plot_cluster_profile_heatmap(cluster_profile: pd.DataFrame):
    features = [f for f in KEY_PROFILE_FEATURES if f in cluster_profile.columns]
    if not features:
        features = cluster_profile.columns[: min(10, len(cluster_profile.columns))].tolist()

    profile = cluster_profile[features]
    profile_z = (profile - profile.mean()) / (profile.std() + 1e-9)

    fig = px.imshow(
        profile_z.T,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-2,
        zmax=2,
        labels=dict(x="Cluster", y="Feature", color="Z-score"),
        title="Cluster Profile Heatmap — Relative Strengths",
    )
    fig.update_layout(height=560, template="plotly_white", title_x=0.02)
    return fig


def plot_radar(cluster_profile: pd.DataFrame):
    features = [f for f in RADAR_FEATURES if f in cluster_profile.columns]
    if len(features) < 3:
        return None

    radar = cluster_profile[features].copy()
    for col in radar.columns:
        radar[col] = (radar[col] - radar[col].min()) / (
            radar[col].max() - radar[col].min() + 1e-9
        )

    fig = go.Figure()
    for cluster_id, row in radar.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=row.tolist() + [row.tolist()[0]],
                theta=features + [features[0]],
                fill="toself",
                name=f"Cluster {cluster_id}",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_white",
        height=620,
        title="Radar Profile — Normalized Cluster Comparison",
    )
    return fig


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, data in sheets.items():
            clean_name = sheet_name[:31]
            data.to_excel(writer, sheet_name=clean_name, index=False)
    return output.getvalue()


def add_engineered_features_for_single_country(row: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for one user-input row."""
    df = row.copy()

    if "GDP" in df.columns and "Population Total" in df.columns:
        df["GDP_per_Capita"] = df["GDP"] / (df["Population Total"] + 1)

    if "Life Expectancy Female" in df.columns and "Life Expectancy Male" in df.columns:
        df["Life_Exp_Gap"] = df["Life Expectancy Female"] - df["Life Expectancy Male"]
        df["Life_Exp_Avg"] = (df["Life Expectancy Female"] + df["Life Expectancy Male"]) / 2

    if "Health Exp % GDP" in df.columns and "Health Exp/Capita" in df.columns:
        df["Health_Investment_Index"] = df["Health Exp % GDP"] * np.log1p(
            df["Health Exp/Capita"].clip(lower=0)
        )

    if "Population 0-14" in df.columns and "Population 65+" in df.columns:
        df["Youthfulness_Index"] = df["Population 0-14"] / (df["Population 65+"] + 0.001)

    if "Internet Usage" in df.columns and "Mobile Phone Usage" in df.columns:
        df["Digital_Access"] = (df["Internet Usage"] + df["Mobile Phone Usage"]) / 2

    if "CO2 Emissions" in df.columns and "Energy Usage" in df.columns:
        df["CO2_Intensity"] = df["CO2 Emissions"] / (df["Energy Usage"] + 1)

    return df.replace([np.inf, -np.inf], np.nan)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🌍 Project Controls")
    st.caption("Upload your original World Development dataset and control the clustering model.")

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload your World_development_measurement Excel/CSV file.",
    )

    st.divider()
    st.markdown("### ⚙️ Clustering Settings")
    k_mode = st.radio(
        "K selection",
        ["Auto best K by Silhouette", "Manual K"],
        index=0,
    )
    manual_k = st.slider("Manual number of clusters", 2, 10, 4)

    model_choice = st.selectbox(
        "Primary model for dashboard",
        ["K-Means", "Agglomerative", "GMM", "DBSCAN"],
        index=0,
    )

    st.divider()
    st.markdown("### 💡 App Features")
    st.write("✅ Data cleaning")
    st.write("✅ Feature engineering")
    st.write("✅ PCA visualization")
    st.write("✅ 4 clustering models")
    st.write("✅ Cluster profiling")
    st.write("✅ Country search")
    st.write("✅ Download outputs")

# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🌍 World Development Clustering Dashboard</div>
        <div class="hero-subtitle">
            A professional unsupervised machine learning app to group countries by development, health,
            demographic, digital, environmental, and economic indicators. Built from your notebook workflow:
            cleaning → aggregation → feature engineering → scaling → PCA → clustering → profiling.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("👈 Upload your Excel/CSV dataset from the sidebar to start the dashboard.")
    st.markdown(
        """
        ### Expected dataset format
        Your file should contain a **Country** column and development indicator columns such as:
        - GDP
        - Birth Rate
        - Internet Usage
        - Life Expectancy Female / Male
        - Infant Mortality Rate
        - Population Urban
        - Health Exp % GDP
        - CO2 Emissions
        """
    )
    st.stop()

# ============================================================
# Data Loading + Model Pipeline
# ============================================================

try:
    df_raw = load_data(uploaded_file)
    is_valid, validation_msg = validate_data(df_raw)
    if not is_valid:
        st.error(validation_msg)
        st.stop()

    with st.spinner("Processing data and training clustering models..."):
        prep = preprocess_data(df_raw)
        X_scaled = prep["X_scaled"]
        df_fe = prep["df_fe"]
        X_pca2 = prep["X_pca2"]
        X_pca3 = prep["X_pca3"]
        pca_2d = prep["pca_2d"]
        scaler = prep["scaler"]
        feature_cols = prep["feature_cols"]

        k_metrics = find_optimal_k(X_scaled, 2, min(10, len(df_fe) - 1))
        auto_k = int(k_metrics.loc[k_metrics["Silhouette"].idxmax(), "K"])
        selected_k = auto_k if k_mode == "Auto best K by Silhouette" else manual_k
        selected_k = min(selected_k, len(df_fe) - 1)

        cluster_results = run_clustering(X_scaled, selected_k)

    label_map = {
        "K-Means": cluster_results["km_labels"],
        "Agglomerative": cluster_results["agg_labels"],
        "GMM": cluster_results["gmm_labels"],
        "DBSCAN": cluster_results["db_labels"],
    }

    cluster_col = f"{model_choice}_Cluster"
    primary_labels = label_map[model_choice]
    df_clustered = build_clustered_df(df_fe, X_pca2, X_pca3, primary_labels, cluster_col)

    # Keep cluster as string for better Plotly category colors
    df_clustered[cluster_col] = df_clustered[cluster_col].astype(str)

    profile_source = df_fe.copy()
    profile_source[cluster_col] = primary_labels
    cluster_profile = profile_source.groupby(cluster_col)[feature_cols].mean()

except Exception as e:
    st.error(f"Something went wrong while processing the project: {e}")
    st.stop()

# ============================================================
# Top Metrics
# ============================================================

countries = df_raw["Country"].nunique()
rows, cols = df_raw.shape
features_after = df_fe.shape[1]
pca_variance = pca_2d.explained_variance_ratio_.sum() * 100
n_clusters = len(set(primary_labels)) - (1 if -1 in set(primary_labels) else 0)
n_noise = int((np.asarray(primary_labels) == -1).sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows", f"{rows:,}")
m2.metric("Countries", f"{countries:,}")
m3.metric("Features Used", f"{features_after:,}")
m4.metric("Selected K", selected_k)
m5.metric("PCA 2D Variance", f"{pca_variance:.1f}%")

if n_noise > 0:
    st.warning(f"{model_choice} detected {n_noise} noise/outlier countries.")

# ============================================================
# Tabs
# ============================================================

tab_overview, tab_eda, tab_cluster, tab_profile, tab_country, tab_predict, tab_download = st.tabs(
    [
        "📌 Overview",
        "🔍 EDA",
        "🤖 Clustering",
        "📊 Profiles",
        "🗺️ Country Explorer",
        "🔮 Predict",
        "⬇️ Downloads",
    ]
)

# ============================================================
# Overview Tab
# ============================================================

with tab_overview:
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">This section summarizes the business goal, workflow, and final modelling setup.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown(
            """
            #### 🎯 Business Goal
            Group countries into meaningful development clusters using economic, health,
            demographic, digital, and environmental indicators.

            #### 🧠 Why Clustering?
            The dataset has no predefined label like **developed**, **developing**, or **low income**.
            Clustering allows the data to naturally form groups based on similar patterns.

            #### 🔁 ML Workflow
            **Load Data → Clean Currency Columns → Aggregate by Country → Impute Missing Values → Feature Engineering → Scaling → PCA → Clustering → Evaluation → Profiling**
            """
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "Step": [
                        "Data Cleaning",
                        "Aggregation",
                        "Feature Engineering",
                        "Scaling",
                        "Dimensionality Reduction",
                        "Clustering",
                        "Evaluation",
                    ],
                    "Technique": [
                        "Currency/percentage conversion",
                        "Mean by country",
                        "GDP per capita, digital access, life expectancy gap, etc.",
                        "StandardScaler",
                        "PCA 2D/3D",
                        "K-Means, Agglomerative, DBSCAN, GMM",
                        "Silhouette, Davies-Bouldin, Calinski-Harabasz",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("#### 📄 Raw Data Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.markdown("#### ✅ Columns Dropped Due to >50% Missing")
        if prep["drop_cols"]:
            st.write(prep["drop_cols"])
        else:
            st.success("No columns were dropped using the >50% missing-value rule.")

# ============================================================
# EDA Tab
# ============================================================

with tab_eda:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Check missing values, distributions, and relationships before clustering.</div>',
        unsafe_allow_html=True,
    )

    eda1, eda2 = st.columns([1, 1])

    with eda1:
        st.markdown("#### Missing Value Analysis")
        fig_missing = plot_missing_values(df_raw)
        if fig_missing is None:
            st.success("No missing values found in the uploaded dataset.")
        else:
            st.plotly_chart(fig_missing, use_container_width=True)

    with eda2:
        st.markdown("#### Feature Distribution")
        numeric_cols = prep["df_proc"].select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Number of Records"]
        selected_feature = st.selectbox("Select feature", numeric_cols)

        fig_hist = px.histogram(
            prep["df_proc"],
            x=selected_feature,
            nbins=35,
            marginal="box",
            title=f"Distribution of {selected_feature}",
            template="plotly_white",
        )
        fig_hist.update_layout(height=470, title_x=0.02)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Correlation Heatmap")
    with st.expander("Show correlation heatmap", expanded=False):
        st.plotly_chart(plot_correlation(df_fe), use_container_width=True)

# ============================================================
# Clustering Tab
# ============================================================

with tab_cluster:
    st.markdown('<div class="section-title">Clustering Model Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Compare models and inspect cluster separation in PCA space.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.15, 0.85])

    with c1:
        st.plotly_chart(
            plot_pca_2d(
                df_clustered,
                cluster_col,
                f"{model_choice} Clusters — PCA 2D View",
            ),
            use_container_width=True,
        )

    with c2:
        st.markdown("#### Model Evaluation")
        eval_df = cluster_results["eval_df"].copy()
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        fig_eval = px.bar(
            eval_df.melt(id_vars="Model", value_vars=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]),
            x="Model",
            y="value",
            color="variable",
            barmode="group",
            title="Evaluation Metrics by Model",
            template="plotly_white",
        )
        fig_eval.update_layout(height=430, title_x=0.02, legend_title_text="Metric")
        st.plotly_chart(fig_eval, use_container_width=True)

    st.markdown("#### K-Means K Selection")
    k1, k2 = st.columns(2)

    with k1:
        fig_sil = px.line(
            k_metrics,
            x="K",
            y="Silhouette",
            markers=True,
            title=f"Silhouette Score by K — Auto Best K = {auto_k}",
            template="plotly_white",
        )
        fig_sil.add_vline(x=auto_k, line_dash="dash", line_color="green")
        fig_sil.update_layout(height=400, title_x=0.02)
        st.plotly_chart(fig_sil, use_container_width=True)

    with k2:
        fig_elbow = px.line(
            k_metrics,
            x="K",
            y="Inertia",
            markers=True,
            title="Elbow Method — Inertia by K",
            template="plotly_white",
        )
        fig_elbow.update_layout(height=400, title_x=0.02)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with st.expander("Show 3D PCA Cluster View", expanded=False):
        st.plotly_chart(
            plot_pca_3d(
                df_clustered,
                cluster_col,
                f"{model_choice} Clusters — PCA 3D View",
            ),
            use_container_width=True,
        )

# ============================================================
# Profiles Tab
# ============================================================

with tab_profile:
    st.markdown('<div class="section-title">Cluster Profiling & Interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Understand what each cluster represents using average feature values.</div>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns([1, 1])

    with p1:
        st.markdown("#### Countries per Cluster")
        size_df = (
            df_clustered.groupby(cluster_col)["Country"]
            .count()
            .reset_index(name="Countries")
            .sort_values(cluster_col)
        )
        fig_size = px.bar(
            size_df,
            x=cluster_col,
            y="Countries",
            text="Countries",
            title="Cluster Size Distribution",
            template="plotly_white",
            color=cluster_col,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_size.update_traces(textposition="outside")
        fig_size.update_layout(height=440, title_x=0.02, showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)

    with p2:
        st.markdown("#### Cluster Meaning Guide")
        st.markdown(
            """
            Use the profile heatmap and radar chart to name clusters:

            - **High Development:** high life expectancy, internet usage, GDP/capita; low infant mortality.
            - **Emerging / Middle Development:** mixed indicators, growing urbanization and moderate digital access.
            - **Low Development:** high birth rate, high infant mortality, lower digital/economic indicators.
            - **Outliers / Noise:** countries with unusual patterns detected by DBSCAN.
            """
        )
        st.dataframe(cluster_profile.round(2), use_container_width=True)

    st.plotly_chart(plot_cluster_profile_heatmap(cluster_profile), use_container_width=True)

    radar_fig = plot_radar(cluster_profile)
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)

# ============================================================
# Country Explorer Tab
# ============================================================

with tab_country:
    st.markdown('<div class="section-title">Country Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Search countries, filter by cluster, and inspect feature values.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        search_text = st.text_input("Search country", "")
    with f2:
        cluster_options = ["All"] + sorted(df_clustered[cluster_col].unique().tolist())
        selected_cluster_filter = st.selectbox("Filter cluster", cluster_options)
    with f3:
        top_n = st.slider("Rows to display", 10, min(208, len(df_clustered)), 50)

    filtered = df_clustered.copy()
    if search_text.strip():
        filtered = filtered[filtered["Country"].str.contains(search_text.strip(), case=False, na=False)]
    if selected_cluster_filter != "All":
        filtered = filtered[filtered[cluster_col] == selected_cluster_filter]

    display_cols = ["Country", cluster_col, "PC1", "PC2"]
    important_cols = [
        "GDP_per_Capita",
        "Life_Exp_Avg",
        "Internet Usage",
        "Infant Mortality Rate",
        "Birth Rate",
        "Population Urban",
        "Digital_Access",
    ]
    display_cols += [c for c in important_cols if c in filtered.columns]

    st.dataframe(filtered[display_cols].head(top_n), use_container_width=True, hide_index=True)

    st.markdown("#### Single Country Deep Dive")
    selected_country = st.selectbox("Select country", df_clustered["Country"].sort_values().tolist())
    country_row = df_clustered[df_clustered["Country"] == selected_country].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Cluster", country_row[cluster_col])
    d2.metric("PC1", f"{country_row['PC1']:.2f}")
    d3.metric("PC2", f"{country_row['PC2']:.2f}")
    if "GDP_per_Capita" in country_row:
        d4.metric("GDP per Capita", f"{country_row['GDP_per_Capita']:,.0f}")
    else:
        d4.metric("Features", len(feature_cols))

    selected_features = [c for c in important_cols if c in df_clustered.columns]
    if selected_features:
        country_values = pd.DataFrame(
            {
                "Feature": selected_features,
                "Value": [country_row[c] for c in selected_features],
            }
        )
        fig_country = px.bar(
            country_values,
            x="Feature",
            y="Value",
            title=f"Key Indicators — {selected_country}",
            template="plotly_white",
        )
        fig_country.update_layout(height=430, title_x=0.02, xaxis_tickangle=-30)
        st.plotly_chart(fig_country, use_container_width=True)

# ============================================================
# Predict Tab
# ============================================================

with tab_predict:
    st.markdown('<div class="section-title">Predict Cluster for a New / Edited Country</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Use current country values as a template, edit metrics, and predict the K-Means cluster.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Prediction uses the trained **K-Means** model because it is the most interpretable deployment model in this project."
    )

    template_country = st.selectbox(
        "Choose a country as starting template",
        df_fe.index.sort_values().tolist(),
        key="predict_template_country",
    )

    base_row = df_fe.loc[[template_country], feature_cols].copy()

    editable_features = [
        c
        for c in [
            "GDP",
            "Population Total",
            "Birth Rate",
            "Life Expectancy Female",
            "Life Expectancy Male",
            "Infant Mortality Rate",
            "Internet Usage",
            "Mobile Phone Usage",
            "Population Urban",
            "Health Exp % GDP",
            "Health Exp/Capita",
            "CO2 Emissions",
            "Energy Usage",
            "Population 0-14",
            "Population 65+",
        ]
        if c in base_row.columns
    ]

    st.markdown("#### Edit Key Inputs")
    input_values = {}
    cols_for_inputs = st.columns(3)
    for idx, feat in enumerate(editable_features):
        current_value = float(base_row.iloc[0][feat])
        with cols_for_inputs[idx % 3]:
            input_values[feat] = st.number_input(
                feat,
                value=current_value,
                format="%.6f",
                key=f"input_{feat}",
            )

    custom_base = base_row.copy()
    for feat, val in input_values.items():
        custom_base.loc[template_country, feat] = val

    # Recalculate engineered values from edited base values, then align columns
    custom_engineered = add_engineered_features_for_single_country(custom_base)
    for col in feature_cols:
        if col not in custom_engineered.columns:
            custom_engineered[col] = df_fe[col].median()
    custom_engineered = custom_engineered[feature_cols].fillna(df_fe.median(numeric_only=True))

    custom_scaled = scaler.transform(custom_engineered)
    pred_cluster = cluster_results["kmeans"].predict(custom_scaled)[0]
    pred_pca = pca_2d.transform(custom_scaled)[0]

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted K-Means Cluster", int(pred_cluster))
    r2.metric("Predicted PC1", f"{pred_pca[0]:.2f}")
    r3.metric("Predicted PC2", f"{pred_pca[1]:.2f}")

    st.markdown("#### Where This Prediction Falls on the Map")
    pred_df = df_clustered.copy()
    pred_df["Point Type"] = "Existing Country"
    new_point = pd.DataFrame(
        {
            "Country": [f"Custom based on {template_country}"],
            cluster_col: [str(pred_cluster)],
            "PC1": [pred_pca[0]],
            "PC2": [pred_pca[1]],
            "PC3": [0],
            "Point Type": ["Prediction"],
        }
    )
    pred_plot_df = pd.concat([pred_df, new_point], ignore_index=True)

    fig_pred = px.scatter(
        pred_plot_df,
        x="PC1",
        y="PC2",
        color=cluster_col,
        symbol="Point Type",
        hover_name="Country",
        title="Prediction Position in PCA Space",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pred.update_traces(marker=dict(size=10, line=dict(width=0.8, color="white")))
    fig_pred.update_layout(height=600, title_x=0.02)
    st.plotly_chart(fig_pred, use_container_width=True)

# ============================================================
# Downloads Tab
# ============================================================

with tab_download:
    st.markdown('<div class="section-title">Download Project Outputs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Export clustered country-level data, cluster profiles, and model evaluation tables.</div>',
        unsafe_allow_html=True,
    )

    clustered_export = df_clustered.drop(columns=["PC1", "PC2", "PC3"], errors="ignore")
    profiles_export = cluster_profile.reset_index()
    eval_export = cluster_results["eval_df"]
    k_metrics_export = k_metrics

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "⬇️ Download Clustered CSV",
            data=convert_df_to_csv(clustered_export),
            file_name="World_Development_Clustered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Download Profiles CSV",
            data=convert_df_to_csv(profiles_export),
            file_name="Cluster_Profiles.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d3:
        excel_bytes = convert_df_to_excel(
            {
                "Clustered_Data": clustered_export,
                "Cluster_Profiles": profiles_export,
                "Model_Evaluation": eval_export,
                "K_Selection": k_metrics_export,
            }
        )
        st.download_button(
            "⬇️ Download Full Excel Report",
            data=excel_bytes,
            file_name="World_Development_Clustering_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("#### Final Output Preview")
    st.dataframe(clustered_export.head(30), use_container_width=True, hide_index=True)

# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Built with Streamlit, Plotly, Pandas, Scikit-learn | Project: World Development Measurement — Cluster Analysis"
)
