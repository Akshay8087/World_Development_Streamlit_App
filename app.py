# ============================================================
# 🌍 World Development Clustering — PKL Deployment Dashboard
# Purpose: Load saved clustering bundle and display direct insights
# ============================================================

import io
import warnings
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="World Development Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).with_name("clustering_pipeline.pkl")

KEY_FEATURES = [
    "GDP_per_Capita",
    "Life_Exp_Avg",
    "Infant Mortality Rate",
    "Internet Usage",
    "Digital_Access",
    "Population Urban",
    "Health Exp/Capita",
    "Birth Rate",
]

RADAR_FEATURES = [
    "GDP_per_Capita",
    "Life_Exp_Avg",
    "Infant Mortality Rate",
    "Digital_Access",
    "Population Urban",
    "Health Exp/Capita",
]

EDITABLE_FEATURES = [
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

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .main {background-color: #f6f8fc;}
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
        .hero-card {
            background: linear-gradient(125deg, #071a34 0%, #0d4d86 58%, #0891b2 100%);
            padding: 32px 34px;
            border-radius: 26px;
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
            margin-bottom: 22px;
        }
        .hero-title {font-size: 43px; font-weight: 800; margin-bottom: 8px; letter-spacing: -1px;}
        .hero-subtitle {font-size: 16.5px; line-height: 1.62; opacity: 0.94; max-width: 1050px;}
        .insight-card {
            background: #ffffff;
            border: 1px solid #e8eef6;
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            min-height: 162px;
            box-shadow: 0 7px 22px rgba(15, 23, 42, 0.05);
        }
        .insight-title {font-size: 18px; font-weight: 750; color: #0f172a; margin-bottom: 8px;}
        .insight-text {font-size: 14px; color: #475569; line-height: 1.48;}
        .section-title {font-size: 26px; font-weight: 800; color: #0f172a; margin: 12px 0 4px 0;}
        .section-caption {color: #526277; margin-bottom: 18px;}
        .pill {
            background: #e0f2fe; color: #075985; border-radius: 999px; padding: 4px 12px;
            font-size: 12px; font-weight: 700; display: inline-block; margin-right: 5px;
        }
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff; border-radius: 14px 14px 0 0;
            border: 1px solid #e5e7eb; padding: 12px 18px;
        }
        .stTabs [aria-selected="true"] {background: #e0f2fe; color: #075985; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load Saved Deployment Bundle
# ============================================================

@st.cache_resource(show_spinner=False)
def load_bundle(model_path: str) -> Dict[str, object]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {path.name}. Keep clustering_pipeline.pkl in the same folder as app.py."
        )
    bundle = joblib.load(path)
    required = ["scaler", "pca_2d", "kmeans", "feature_columns", "optimal_k", "cluster_profile"]
    missing = [item for item in required if item not in bundle]
    if missing:
        raise ValueError(f"Saved bundle is missing required objects: {', '.join(missing)}")
    return bundle


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if {"GDP", "Population Total"}.issubset(result.columns):
        result["GDP_per_Capita"] = result["GDP"] / (result["Population Total"] + 1)

    if {"Life Expectancy Female", "Life Expectancy Male"}.issubset(result.columns):
        result["Life_Exp_Gap"] = result["Life Expectancy Female"] - result["Life Expectancy Male"]
        result["Life_Exp_Avg"] = (
            result["Life Expectancy Female"] + result["Life Expectancy Male"]
        ) / 2

    if {"Health Exp % GDP", "Health Exp/Capita"}.issubset(result.columns):
        result["Health_Investment_Index"] = result["Health Exp % GDP"] * np.log1p(
            result["Health Exp/Capita"].clip(lower=0)
        )

    if {"Population 0-14", "Population 65+"}.issubset(result.columns):
        result["Youthfulness_Index"] = result["Population 0-14"] / (
            result["Population 65+"] + 0.001
        )

    if {"Internet Usage", "Mobile Phone Usage"}.issubset(result.columns):
        result["Digital_Access"] = (
            result["Internet Usage"] + result["Mobile Phone Usage"]
        ) / 2

    if {"CO2 Emissions", "Energy Usage"}.issubset(result.columns):
        result["CO2_Intensity"] = result["CO2 Emissions"] / (result["Energy Usage"] + 1)

    return result.replace([np.inf, -np.inf], np.nan)


def display_name(cluster_id: object, profile: pd.DataFrame) -> str:
    cluster_id = int(cluster_id)
    if "GDP_per_Capita" in profile.columns and "Life_Exp_Avg" in profile.columns:
        low_gdp = profile["GDP_per_Capita"].idxmin()
        high_gdp = profile["GDP_per_Capita"].idxmax()
        if cluster_id == int(low_gdp):
            return "Developing / High-Need"
        if cluster_id == int(high_gdp):
            return "High-Income / Advanced"
    return "Progressing / Transitional"


def narrative(cluster_id: int, profile: pd.DataFrame) -> str:
    row = profile.loc[cluster_id]
    name = display_name(cluster_id, profile)
    parts: List[str] = []

    if "GDP_per_Capita" in row:
        parts.append(f"GDP/capita ≈ {row['GDP_per_Capita']:,.0f}")
    if "Life_Exp_Avg" in row:
        parts.append(f"life expectancy ≈ {row['Life_Exp_Avg']:.1f} years")
    if "Internet Usage" in row:
        parts.append(f"internet usage ≈ {row['Internet Usage']:.1%}")
    if "Infant Mortality Rate" in row:
        parts.append(f"infant mortality ≈ {row['Infant Mortality Rate']:.1%}")

    return f"<b>{name}</b><br>{' · '.join(parts)}"


def normalised_profile(profile: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    use = [col for col in features if col in profile.columns]
    result = profile[use].copy()
    return (result - result.min()) / (result.max() - result.min() + 1e-9)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


try:
    bundle = load_bundle(str(MODEL_PATH))
    scaler = bundle["scaler"]
    pca_2d = bundle["pca_2d"]
    kmeans = bundle["kmeans"]
    feature_cols = list(bundle["feature_columns"])
    optimal_k = int(bundle["optimal_k"])
    cluster_profile = bundle["cluster_profile"].copy()
    cluster_profile.index = cluster_profile.index.astype(int)
    cluster_profile.index.name = "Cluster"
except Exception as exc:
    st.error(f"Unable to load the saved deployment pipeline: {exc}")
    st.info("Place `clustering_pipeline.pkl` beside this app file and use compatible package versions from requirements.txt.")
    st.stop()

# Create direct insight data
profile_display = cluster_profile.copy()
profile_display["Segment Name"] = [display_name(idx, cluster_profile) for idx in profile_display.index]

centroids_scaled = scaler.transform(cluster_profile[feature_cols])
centroids_pca = pca_2d.transform(centroids_scaled)
centroid_df = pd.DataFrame(
    {
        "PC1": centroids_pca[:, 0],
        "PC2": centroids_pca[:, 1],
        "Cluster": cluster_profile.index.astype(str),
        "Segment": [display_name(idx, cluster_profile) for idx in cluster_profile.index],
    }
)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 🌍 Intelligence Console")
    st.success("Saved pipeline loaded successfully")
    st.caption("No dataset upload required. Insights are read directly from your trained `.pkl` bundle.")

    st.divider()
    st.markdown("### 📦 Deployment Objects")
    st.write("✅ Median Imputer" if "imputer" in bundle else "➖ Imputer unavailable")
    st.write("✅ Standard Scaler")
    st.write("✅ PCA projection")
    st.write("✅ K-Means model")
    st.write("✅ Cluster profile table")
    if "gmm" in bundle:
        st.write("✅ GMM model")
    if "dbscan" in bundle:
        st.write("✅ DBSCAN model")
    if "agg" in bundle:
        st.write("✅ Agglomerative model")

    st.divider()
    selected_cluster = st.selectbox(
        "Focus cluster",
        cluster_profile.index.tolist(),
        format_func=lambda x: f"Cluster {x} — {display_name(x, cluster_profile)}",
    )

# ============================================================
# Header & KPI Cards
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🌍 World Development Intelligence</div>
        <div class="hero-subtitle">
            Deployment-ready clustering dashboard powered directly by the saved machine learning pipeline.
            Explore the three discovered development segments, compare their economic and social signatures,
            and classify a new country profile without re-uploading or retraining the dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Optimal Clusters", optimal_k)
m2.metric("Features Used", len(feature_cols))
m3.metric("Deployment Model", "K-Means")
m4.metric("PCA Components", getattr(pca_2d, "n_components_", 2))

tab_overview, tab_insights, tab_visuals, tab_predict, tab_export = st.tabs(
    ["📌 Executive View", "📊 Cluster Insights", "🗺️ Visual Analytics", "🔮 Predict Segment", "⬇️ Export"]
)

# ============================================================
# Executive View
# ============================================================

with tab_overview:
    st.markdown('<div class="section-title">Executive Cluster Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Instant insights are generated from the cluster profiles stored inside your trained model bundle.</div>',
        unsafe_allow_html=True,
    )

    cards = st.columns(optimal_k)
    for i, cluster_id in enumerate(cluster_profile.index):
        with cards[i % len(cards)]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="pill">Cluster {cluster_id}</div>
                    <div class="insight-title">{display_name(cluster_id, cluster_profile)}</div>
                    <div class="insight-text">{narrative(cluster_id, cluster_profile)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### 🎯 Business Interpretation")
    lower_id = int(cluster_profile["GDP_per_Capita"].idxmin()) if "GDP_per_Capita" in cluster_profile else 0
    upper_id = int(cluster_profile["GDP_per_Capita"].idxmax()) if "GDP_per_Capita" in cluster_profile else optimal_k - 1
    st.markdown(
        f"""
        - **Cluster {lower_id}** needs policy focus on health, infrastructure, connectivity, and income growth.
        - **Cluster {upper_id}** represents stronger economic and human-development performance and can be benchmarked for best practices.
        - The remaining cluster represents countries progressing through digital, urban, and economic transition.
        """
    )

    selected_cols = [c for c in KEY_FEATURES if c in cluster_profile.columns]
    st.dataframe(
        profile_display[["Segment Name"] + selected_cols].round(3),
        use_container_width=True,
    )

# ============================================================
# Cluster Insights
# ============================================================

with tab_insights:
    name = display_name(selected_cluster, cluster_profile)
    st.markdown(f'<div class="section-title">Cluster {selected_cluster}: {name}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Inspect the average indicators characterising this segment.</div>',
        unsafe_allow_html=True,
    )

    row = cluster_profile.loc[selected_cluster]
    metric_features = [f for f in KEY_FEATURES if f in row.index][:8]
    metric_cols = st.columns(4)
    for i, feature in enumerate(metric_features):
        value = float(row[feature])
        if feature in ["GDP_per_Capita", "Health Exp/Capita"]:
            shown = f"{value:,.0f}"
        elif feature in ["Internet Usage", "Digital_Access", "Infant Mortality Rate", "Birth Rate", "Population Urban"]:
            shown = f"{value:.1%}"
        else:
            shown = f"{value:.2f}"
        metric_cols[i % 4].metric(feature, shown)

    st.markdown("#### Relative Feature Position")
    usable = [col for col in KEY_FEATURES if col in cluster_profile.columns]
    norm = normalised_profile(cluster_profile, usable).loc[selected_cluster].reset_index()
    norm.columns = ["Feature", "Relative Score"]
    fig_selected = px.bar(
        norm,
        x="Relative Score",
        y="Feature",
        orientation="h",
        range_x=[0, 1],
        template="plotly_white",
        title=f"Cluster {selected_cluster} Compared with Other Segments",
    )
    fig_selected.update_layout(height=470, title_x=0.02, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_selected, use_container_width=True)

# ============================================================
# Visual Analytics
# ============================================================

with tab_visuals:
    st.markdown('<div class="section-title">Cluster Comparison Visuals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">These views compare saved cluster centroids rather than individual countries because country-level output was not packaged in the current bundle.</div>',
        unsafe_allow_html=True,
    )

    cols_heat = [c for c in KEY_FEATURES if c in cluster_profile.columns]
    z_profile = (
        (cluster_profile[cols_heat] - cluster_profile[cols_heat].mean())
        / (cluster_profile[cols_heat].std() + 1e-9)
    )
    fig_heat = px.imshow(
        z_profile.T,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-2,
        zmax=2,
        labels=dict(x="Cluster", y="Indicator", color="Relative score"),
        title="Cluster Signature Heatmap",
    )
    fig_heat.update_layout(height=560, template="plotly_white", title_x=0.02)
    st.plotly_chart(fig_heat, use_container_width=True)

    v1, v2 = st.columns([1.05, 0.95])
    with v1:
        radar_features = [c for c in RADAR_FEATURES if c in cluster_profile.columns]
        radar_df = normalised_profile(cluster_profile, radar_features)
        fig_radar = go.Figure()
        for cluster_id, values in radar_df.iterrows():
            labels = radar_features + [radar_features[0]]
            r_values = values.tolist() + [values.tolist()[0]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=labels,
                    fill="toself",
                    name=f"Cluster {cluster_id}: {display_name(cluster_id, cluster_profile)}",
                )
            )
        fig_radar.update_layout(
            title="Normalized Segment Radar",
            template="plotly_white",
            height=570,
            polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with v2:
        fig_pca = px.scatter(
            centroid_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            hover_name="Segment",
            text="Segment",
            title="PCA Map of Saved Cluster Centroids",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pca.update_traces(marker=dict(size=20, line=dict(width=1, color="white")), textposition="top center")
        fig_pca.update_layout(height=570, title_x=0.02)
        st.plotly_chart(fig_pca, use_container_width=True)

# ============================================================
# Prediction
# ============================================================

with tab_predict:
    st.markdown('<div class="section-title">Predict a Development Segment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Begin from a saved cluster profile, edit the inputs, and send the engineered feature vector through the stored scaler and K-Means model.</div>',
        unsafe_allow_html=True,
    )

    template_cluster = st.selectbox(
        "Starting template",
        cluster_profile.index.tolist(),
        format_func=lambda x: f"Cluster {x} — {display_name(x, cluster_profile)}",
        key="prediction_template",
    )
    custom = cluster_profile.loc[[template_cluster], feature_cols].copy()
    editable = [feature for feature in EDITABLE_FEATURES if feature in custom.columns]

    with st.form("prediction_form"):
        input_cols = st.columns(3)
        entered = {}
        for i, feature in enumerate(editable):
            with input_cols[i % 3]:
                entered[feature] = st.number_input(
                    feature,
                    value=float(custom.iloc[0][feature]),
                    format="%.6f",
                )
        submitted = st.form_submit_button("Predict Development Cluster", type="primary", use_container_width=True)

    if submitted:
        for feature, value in entered.items():
            custom.loc[template_cluster, feature] = value

        custom = add_engineered_features(custom)
        custom = custom.reindex(columns=feature_cols)
        custom = custom.fillna(cluster_profile[feature_cols].median(numeric_only=True))

        transformed = scaler.transform(custom)
        predicted_cluster = int(kmeans.predict(transformed)[0])
        pred_pca = pca_2d.transform(transformed)[0]
        predicted_name = display_name(predicted_cluster, cluster_profile)

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Cluster", predicted_cluster)
        r2.metric("Segment", predicted_name)
        r3.metric("PCA Position", f"{pred_pca[0]:.2f}, {pred_pca[1]:.2f}")

        prediction_df = centroid_df.copy()
        prediction_point = pd.DataFrame(
            {
                "PC1": [pred_pca[0]],
                "PC2": [pred_pca[1]],
                "Cluster": [str(predicted_cluster)],
                "Segment": ["New Prediction"],
                "Point Type": ["Predicted Country"],
            }
        )
        prediction_df["Point Type"] = "Saved Cluster Centroid"
        prediction_df = pd.concat([prediction_df, prediction_point], ignore_index=True)

        fig_prediction = px.scatter(
            prediction_df,
            x="PC1",
            y="PC2",
            color="Cluster",
            symbol="Point Type",
            hover_name="Segment",
            title="Prediction Position Versus Cluster Centroids",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_prediction.update_traces(marker=dict(size=16, line=dict(width=1, color="white")))
        fig_prediction.update_layout(height=560, title_x=0.02)
        st.plotly_chart(fig_prediction, use_container_width=True)

# ============================================================
# Export
# ============================================================

with tab_export:
    st.markdown('<div class="section-title">Export Stored Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Download the cluster profile output already saved within the deployment bundle.</div>',
        unsafe_allow_html=True,
    )

    export_profile = profile_display.copy()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button(
            "⬇️ Download Cluster Profiles CSV",
            data=csv_bytes(export_profile),
            file_name="saved_cluster_profiles.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.info(
            "For individual country explorer, country counts, and evaluation charts, save `df_clustered`, "
            "`eval_df`, and `k_metrics` into the PKL file during notebook training."
        )
    st.dataframe(export_profile.round(3), use_container_width=True)

st.divider()
st.caption(
    "Built with Streamlit, Plotly and Scikit-learn | Deployment Mode: saved clustering_pipeline.pkl loaded automatically"
)
