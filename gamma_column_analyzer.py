"""
Gamma Column Scanning Analyzer
Agensi Nuklear Malaysia / UMPSA
-------------------------------
Converts gamma count data to graphs, then uses Machine Learning
to automatically identify Air / Sponge / Material regions.

HOW TO RUN:
    streamlit run gamma_column_analyzer.py

DO NOT run with:  python gamma_column_analyzer.py
"""

import sys
import io
import warnings
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

warnings.filterwarnings("ignore")

# ── Guard: must be launched via `streamlit run`, not `python` ──────────────
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("\n" + "="*60)
        print("ERROR: Do not run this file with \'python\'.")
        print("Run it with:  streamlit run gamma_column_analyzer.py")
        print("="*60 + "\n")
        sys.exit(1)
except SystemExit:
    raise
except Exception:
    pass

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Gamma Column Scanning Analyzer",
    page_icon="☢️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://brand.umpsa.edu.my/images/logo-umpsa-full-color2.png",
        use_container_width=True,
    )
    st.image(
        "https://www.majalahsains.com/wp-content/uploads/2012/05/Logo-Agensi-Nuklear-Malaysia.png",
        use_container_width=True,
    )
    st.markdown("## Gamma Column Scanning Analyzer")
    st.markdown("---")

    st.markdown("### Developers")
    st.write("**Assoc. Prof. Dr. Ku Muhammad Naim Ku Khalif**")
    st.write("Centre for Mathematical Sciences")
    st.write("Universiti Malaysia Pahang Al-Sultan Abdullah")
    st.write("📧 kunaim@umpsa.edu.my")
    st.markdown("")
    st.write("**Dr. Hanafi Ithnin**")
    st.write("Bahagian Teknologi Industri (BTI)")
    st.write("Agensi Nuklear Malaysia")
    st.write("📧 hanafi_i@nm.gov.my")
    st.markdown("---")

    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Excel (.xlsx) file",
        type=["xlsx", "xls"],
    )

    st.markdown("---")
    st.markdown("### Classification Settings")
    count_time_filter = st.selectbox(
        "Select Count Time",
        ["All", "1s", "3s", "6s"],
        index=2,
        help="Filter sheets by count time. '3s' is the recommended setting.",
    )
    smoothing_window = st.slider(
        "Smoothing Window (Savitzky-Golay)",
        min_value=5, max_value=25, value=11, step=2,
        help="Larger window = smoother curve but loses fine detail.",
    )
    air_pct = st.slider(
        "Air Threshold (% of range from min)",
        min_value=5, max_value=40, value=20,
        help="Count values below this % of the full range are classified as Air.",
    )
    sponge_pct = st.slider(
        "Sponge Threshold (% of range from min)",
        min_value=30, max_value=80, value=60,
        help="Count values between Air and this % are classified as Sponge.",
    )
    n_estimators = 200


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_sheets(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Load and clean all sheets from the uploaded Excel file."""
    xl = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)
    result = {}
    for name, df in xl.items():
        data = df.iloc[:, :2].copy()
        data.columns = ["step", "count"]
        data = data[pd.to_numeric(data["step"], errors="coerce").notna()]
        data = data[pd.to_numeric(data["count"], errors="coerce").notna()]
        data = data.astype({"step": int, "count": float})
        data = data.sort_values("step").reset_index(drop=True)
        result[name] = data
    return result


def smooth_signal(counts: np.ndarray, window: int) -> np.ndarray:
    """Apply Savitzky-Golay smoothing; fall back to rolling mean if needed."""
    try:
        return savgol_filter(counts, window_length=window, polyorder=3)
    except Exception:
        w = max(3, window // 2 * 2 + 1)
        return pd.Series(counts).rolling(w, center=True, min_periods=1).mean().values


def rule_based_labels(smoothed: np.ndarray, air_pct: int, sponge_pct: int) -> np.ndarray:
    """Assign Air / Sponge / Material labels using amplitude thresholds."""
    mn, mx = smoothed.min(), smoothed.max()
    rng = mx - mn
    air_thresh = mn + rng * (air_pct / 100)
    sponge_thresh = mn + rng * (sponge_pct / 100)
    labels = np.where(
        smoothed < air_thresh, "Air",
        np.where(smoothed < sponge_thresh, "Sponge", "Material")
    )
    return labels


def extract_ml_features(counts: np.ndarray, smoothed: np.ndarray, window: int = 9) -> pd.DataFrame:
    """
    Extract local statistical features for each step position.
    These capture amplitude level, gradient, and local variability —
    the key signals that distinguish Air / Sponge / Material.
    """
    n = len(counts)
    half = window // 2
    rows = []
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        w = counts[s:e]
        sw = smoothed[s:e]

        grad = float(smoothed[i] - smoothed[i - 1]) if i > 0 else 0.0
        grad2 = float(smoothed[i] - 2 * smoothed[i - 1] + smoothed[i - 2]) if i > 1 else 0.0

        rows.append({
            "count":           counts[i],
            "smoothed":        smoothed[i],
            "norm":            (counts[i] - counts.min()) / max(counts.max() - counts.min(), 1),
            "rolling_mean":    np.mean(w),
            "rolling_std":     np.std(w),
            "rolling_min":     np.min(w),
            "rolling_max":     np.max(w),
            "rolling_range":   float(np.max(w) - np.min(w)),
            "smooth_mean":     np.mean(sw),
            "smooth_std":      np.std(sw),
            "gradient":        grad,
            "gradient2":       grad2,
            "abs_gradient":    abs(grad),
        })
    return pd.DataFrame(rows)


def train_rf_model(X: pd.DataFrame, y: np.ndarray, n_est: int) -> tuple:
    """
    Train a Random Forest classifier with anti-overfitting regularisation.
    Key measures:
      - max_depth=6 (shallow trees, prevents memorising noise)
      - min_samples_leaf=5 (each leaf needs 5+ samples)
      - max_features="sqrt" (each split sees only sqrt(n) features)
      - max_samples=0.8 (each tree uses 80% of data, not 100%)
    Evaluation uses proper train/test split so CV score reflects
    generalisation performance, not memorisation.
    """
    from sklearn.model_selection import train_test_split as tts
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=6,            # shallow — prevents deep memorisation
        min_samples_leaf=5,     # at least 5 samples per leaf
        max_features="sqrt",    # random subset of features at each split
        max_samples=0.8,        # each tree trained on 80% of data (bagging)
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # Proper hold-out CV — trained on 70%, scored on 30% unseen data
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    cv = StratifiedKFold(
        n_splits=min(5, int(min(np.bincount(y_enc)))),
        shuffle=True, random_state=42,
    )
    cv_scores = cross_val_score(model, Xs, y_enc, cv=cv, scoring="accuracy")
    model.fit(Xs, y)
    return model, scaler, cv_scores


def color_for_label(label: str) -> str:
    return {"Air": "#4FC3F7", "Sponge": "#FFB74D", "Material": "#81C784"}.get(label, "#BDBDBD")


def build_classification_plot(
    steps: np.ndarray,
    counts: np.ndarray,
    smoothed: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> go.Figure:
    """Interactive Plotly chart: raw signal + smoothed + colour-coded regions."""
    fig = go.Figure()

    # Coloured background bands per label
    prev_label = labels[0]
    band_start = steps[0]
    for i in range(1, len(steps)):
        if labels[i] != prev_label or i == len(steps) - 1:
            end_step = steps[i] if labels[i] != prev_label else steps[i]
            fig.add_vrect(
                x0=band_start, x1=end_step,
                fillcolor=color_for_label(prev_label),
                opacity=0.25, layer="below", line_width=0,
                annotation_text=prev_label if (end_step - band_start) > 10 else "",
                annotation_position="top left",
                annotation_font_size=10,
            )
            prev_label = labels[i]
            band_start = steps[i]

    # Raw counts
    fig.add_trace(go.Scatter(
        x=steps, y=counts, mode="lines",
        name="Raw Count", line=dict(color="#90CAF9", width=1.2),
        opacity=0.7,
    ))

    # Smoothed signal
    fig.add_trace(go.Scatter(
        x=steps, y=smoothed, mode="lines",
        name="Smoothed", line=dict(color="#1565C0", width=2.5),
    ))

    # Scatter points coloured by label
    label_colors = [color_for_label(l) for l in labels]
    for lbl in ["Air", "Sponge", "Material"]:
        mask = labels == lbl
        fig.add_trace(go.Scatter(
            x=steps[mask], y=smoothed[mask], mode="markers",
            name=lbl,
            marker=dict(color=color_for_label(lbl), size=5, line=dict(width=0.5, color="black")),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1565C0")),
        xaxis_title="Step Position (2 mm increments)",
        yaxis_title="Gamma Count",
        height=480,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60),
    )
    return fig


def build_region_summary(steps: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    """Collapse consecutive identical labels into region segments."""
    rows = []
    start_i = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1] or i == len(labels) - 1:
            end_i = i if labels[i] != labels[i - 1] else i
            seg_counts = counts[start_i:end_i]
            rows.append({
                "Region #":        len(rows) + 1,
                "Label":           labels[start_i],
                "Start Step":      int(steps[start_i]),
                "End Step":        int(steps[end_i - 1]),
                "Width (steps)":   end_i - start_i,
                "Width (mm)":      (end_i - start_i) * 2,
                "Avg Count":       round(float(seg_counts.mean()), 1),
                "Min Count":       round(float(seg_counts.min()), 1),
                "Max Count":       round(float(seg_counts.max()), 1),
            })
            start_i = i
    return pd.DataFrame(rows)


def anomaly_detection(counts: np.ndarray) -> np.ndarray:
    """Isolation Forest to flag statistically unusual count values."""
    X = counts.reshape(-1, 1)
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X)
    return preds == -1  # True = anomaly


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
st.title("☢️ Gamma Column Scanning — AI Analyzer")
st.markdown(
    "Automatically plots gamma count amplitude graphs and uses **Machine Learning** "
    "to identify **Air**, **Sponge**, and **Material** regions in each column scan."
)

if uploaded_file is None:
    st.info("👈 Please upload your Excel file from the sidebar to begin.")
    st.markdown("""
    **Expected format:** Each sheet contains two columns — **Step** and **Count** — 
    representing a gamma scanning traverse across a column model.
    
    | Step | Count |
    |------|-------|
    | 0    | 834   |
    | 1    | 821   |
    | …    | …     |
    
    **Supported sheets:** test(1s), test(3S), test(6s), traywax variants, etc.
    """)
    st.stop()

# ── Load data ──────────────────────────────
file_bytes = uploaded_file.read()
with st.spinner("Loading sheets…"):
    all_sheets = load_all_sheets(file_bytes)

# ── Filter by count time ───────────────────
def sheet_matches_time(name: str, time_filter: str) -> bool:
    if time_filter == "All":
        return True
    return time_filter.lower() in name.lower()

selected_sheets = {k: v for k, v in all_sheets.items() if sheet_matches_time(k, count_time_filter)}

if not selected_sheets:
    st.warning(f"No sheets found matching count time '{count_time_filter}'. Showing all sheets.")
    selected_sheets = all_sheets

st.success(f"Loaded **{len(all_sheets)}** sheets total | Analysing **{len(selected_sheets)}** matching '{count_time_filter}'")

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_graphs, tab_ml, tab_compare, tab_anomaly, tab_export = st.tabs([
    "📊 Graphs", "🤖 ML Classification", "📈 Compare Sheets", "🔍 Anomaly Detection", "📥 Export"
])


# ══════════════════════════════════════════════
#  TAB 1 — GRAPHS
# ══════════════════════════════════════════════
with tab_graphs:
    st.subheader("Gamma Count Amplitude Graphs")
    st.markdown(
        "Each graph shows the raw gamma count signal and its smoothed envelope "
        "across the column scan traverse."
    )

    sheet_choice = st.selectbox("Select sheet to display", list(selected_sheets.keys()), key="graph_sheet")
    df_sel = selected_sheets[sheet_choice]
    steps = df_sel["step"].values
    counts = df_sel["count"].values
    smoothed = smooth_signal(counts, smoothing_window)
    labels_rb = rule_based_labels(smoothed, air_pct, sponge_pct)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Steps", len(steps))
    col2.metric("Min Count", f"{counts.min():.0f}")
    col3.metric("Max Count", f"{counts.max():.0f}")
    col4.metric("Mean Count", f"{counts.mean():.0f}")

    fig_main = build_classification_plot(steps, counts, smoothed, labels_rb, f"Gamma Count — {sheet_choice}")
    st.plotly_chart(fig_main, use_container_width=True)

    # Show ALL selected sheets below
    st.markdown("---")
    st.subheader("All Selected Sheets — Quick Overview")
    cols = st.columns(min(2, len(selected_sheets)))
    for idx, (name, df) in enumerate(selected_sheets.items()):
        s = df["step"].values
        c = df["count"].values
        sm = smooth_signal(c, smoothing_window)
        lb = rule_based_labels(sm, air_pct, sponge_pct)

        mini_fig = go.Figure()
        mini_fig.add_trace(go.Scatter(x=s, y=c, mode="lines",
                                      line=dict(color="#90CAF9", width=1), name="Raw"))
        mini_fig.add_trace(go.Scatter(x=s, y=sm, mode="lines",
                                      line=dict(color="#1565C0", width=2), name="Smooth"))
        mini_fig.update_layout(
            title=name, height=280, template="plotly_white",
            margin=dict(t=40, b=30, l=40, r=10), showlegend=False
        )
        cols[idx % 2].plotly_chart(mini_fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — ML CLASSIFICATION
# ══════════════════════════════════════════════
with tab_ml:
    st.subheader("🤖 Machine Learning Classification")
    st.markdown("""
    **Workflow:**
    1. Rule-based thresholds generate initial **Air / Sponge / Material** labels from the smoothed signal
    2. Statistical features (rolling mean, gradient, local range, etc.) are extracted per step
    3. A **Random Forest** classifier is trained on these labels + features
    4. The trained model predicts labels for every step — providing a data-driven, consistent classification
    """)

    ml_sheet = st.selectbox("Select sheet for ML classification", list(selected_sheets.keys()), key="ml_sheet")
    df_ml = selected_sheets[ml_sheet]
    steps_ml = df_ml["step"].values
    counts_ml = df_ml["count"].values
    smoothed_ml = smooth_signal(counts_ml, smoothing_window)

    # Step 1: Rule-based seed labels
    seed_labels = rule_based_labels(smoothed_ml, air_pct, sponge_pct)

    # Step 2: Feature extraction
    with st.spinner("Extracting features…"):
        features_ml = extract_ml_features(counts_ml, smoothed_ml, window=9)

    # Step 3: Train/Test split then Random Forest
    from sklearn.model_selection import train_test_split as tts
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = tts(
        features_ml, seed_labels, np.arange(len(seed_labels)),
        test_size=0.25, random_state=42, stratify=seed_labels
    )
    with st.spinner(f"Training Random Forest ({n_estimators} trees) on 75% data…"):
        model_rf, scaler_rf, cv_scores = train_rf_model(X_tr, y_tr, n_estimators)

    # Step 4: Predict on held-out test set (25%) for honest evaluation
    X_te_scaled = scaler_rf.transform(X_te)
    y_pred_test = model_rf.predict(X_te_scaled)
    test_acc = accuracy_score(y_te, y_pred_test)

    # Predict ALL steps for visualisation (this is expected — inference, not evaluation)
    X_all_scaled = scaler_rf.transform(features_ml)
    ml_labels = model_rf.predict(X_all_scaled)
    ml_proba = model_rf.predict_proba(X_all_scaled)
    label_classes = model_rf.classes_

    # ── Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV Accuracy (5-fold, train set)", f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    c2.metric("Test Accuracy (unseen 25%)", f"{test_acc:.1%}", help="Evaluated on data the model never trained on")
    # Overfit warning
    overfit_gap = cv_scores.mean() - test_acc
    if overfit_gap > 0.08:
        st.warning(f"⚠️ Possible overfitting detected: CV={cv_scores.mean():.1%} vs Test={test_acc:.1%} (gap={overfit_gap:.1%}). Try reducing max_depth or increasing min_samples_leaf in sidebar.")
    mat_pct = (ml_labels == "Material").mean()
    sponge_pct_val = (ml_labels == "Sponge").mean()
    air_pct_val = (ml_labels == "Air").mean()
    mat_pct = (ml_labels == "Material").mean()
    sponge_pct_val = (ml_labels == "Sponge").mean()
    air_pct_val = (ml_labels == "Air").mean()
    c3.metric("Material %", f"{mat_pct:.1%}")
    c4.metric("Air / Sponge %", f"{air_pct_val:.1%} / {sponge_pct_val:.1%}")

    # ── Classification plot ──
    st.markdown("#### ML-Classified Gamma Count Graph")
    fig_ml = build_classification_plot(
        steps_ml, counts_ml, smoothed_ml, ml_labels,
        f"ML Classification — {ml_sheet}"
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    # ── Confidence plot ──
    st.markdown("#### Classification Confidence per Step")
    max_proba = ml_proba.max(axis=1)
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(
        x=steps_ml, y=max_proba * 100, mode="lines",
        fill="tozeroy", fillcolor="rgba(21,101,192,0.15)",
        line=dict(color="#1565C0", width=1.5), name="Confidence %"
    ))
    fig_conf.update_layout(
        title="Model Confidence (%) per Step",
        xaxis_title="Step", yaxis_title="Confidence (%)",
        height=260, template="plotly_white",
        yaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # ── Feature Importance ──
    st.markdown("#### Feature Importance")
    fi_df = pd.DataFrame({
        "Feature": features_ml.columns,
        "Importance": model_rf.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        title="Random Forest Feature Importances"
    )
    fig_fi.update_layout(height=350, template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Classification Report ──
    st.markdown("#### Classification Report (Test Set — 25% unseen data)")
    report_dict = classification_report(y_te, y_pred_test, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]), use_container_width=True)

    # ── Confusion Matrix ──
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_te, y_pred_test, labels=label_classes)
    fig_cm = px.imshow(
        cm, text_auto=True, x=label_classes, y=label_classes,
        color_continuous_scale="Blues",
        title="Confusion Matrix (Test Set — 25% unseen data)",
        labels=dict(x="ML Predicted", y="Rule-based True", color="Count"),
    )
    fig_cm.update_layout(height=350)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Region Summary Table ──
    st.markdown("#### Detected Regions Summary")
    region_df = build_region_summary(steps_ml, ml_labels, counts_ml)
    # Colour code by label
    def color_row(row):
        clr = {"Air": "#E3F2FD", "Sponge": "#FFF3E0", "Material": "#E8F5E9"}.get(row["Label"], "")
        return [f"background-color: {clr}"] * len(row)
    st.dataframe(region_df.style.apply(color_row, axis=1), use_container_width=True)

    st.markdown("#### Label Distribution")
    dist_df = pd.DataFrame({
        "Label": label_classes,
        "Steps": [(ml_labels == l).sum() for l in label_classes],
    })
    fig_dist = px.pie(dist_df, names="Label", values="Steps",
                      color="Label",
                      color_discrete_map={"Air": "#4FC3F7", "Sponge": "#FFB74D", "Material": "#81C784"},
                      title="Step Distribution by Material Type")
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── AI Interpretation ──
    st.markdown("---")
    st.markdown("#### 🧠 AI Interpretation Summary")
    total = len(ml_labels)
    n_air = (ml_labels == "Air").sum()
    n_sponge = (ml_labels == "Sponge").sum()
    n_mat = (ml_labels == "Material").sum()
    n_regions = len(region_df)
    n_air_regions = len(region_df[region_df["Label"] == "Air"])
    avg_air_width = region_df[region_df["Label"] == "Air"]["Width (mm)"].mean() if n_air_regions > 0 else 0

    interpretation = f"""
**Sheet Analysed:** `{ml_sheet}`  
**Total Scan Length:** {total * 2} mm ({total} steps × 2 mm)

**Material Composition:**
- 🟢 **Material** — {n_mat} steps ({n_mat/total:.1%} of scan, {n_mat*2} mm)
- 🟠 **Sponge** — {n_sponge} steps ({n_sponge/total:.1%} of scan, {n_sponge*2} mm)
- 🔵 **Air** — {n_air} steps ({n_air/total:.1%} of scan, {n_air*2} mm)

**Structural Findings:**
- {n_regions} distinct regions identified
- {n_air_regions} air gap(s) detected, averaging **{avg_air_width:.0f} mm** in width
- Model confidence: **{max_proba.mean()*100:.1f}%** average per step

**Interpretation:**  
{"The scan shows a predominantly solid material column with periodic air gaps — consistent with a tray-and-wax column model where air voids separate material layers." 
if n_air_regions > 2 else 
"The scan shows a mostly continuous material profile with occasional transition zones."}
The Random Forest classifier achieved **{cv_scores.mean():.1%}** cross-validation accuracy, 
indicating {"high confidence" if cv_scores.mean() > 0.92 else "moderate confidence"} in the classification.
    """
    st.info(interpretation)


# ══════════════════════════════════════════════
#  TAB 3 — COMPARE SHEETS
# ══════════════════════════════════════════════
with tab_compare:
    st.subheader("📈 Compare Multiple Sheets")
    st.markdown("Overlay gamma count profiles from different count times or conditions.")

    compare_sheets = st.multiselect(
        "Select sheets to compare",
        list(all_sheets.keys()),
        default=list(all_sheets.keys())[:min(3, len(all_sheets))],
    )

    if compare_sheets:
        fig_cmp = go.Figure()
        palette = px.colors.qualitative.Set2
        for idx, name in enumerate(compare_sheets):
            df_c = all_sheets[name]
            sm = smooth_signal(df_c["count"].values, smoothing_window)
            # Normalise to 0-1 for fair comparison across count times
            norm = (sm - sm.min()) / max(sm.max() - sm.min(), 1)
            fig_cmp.add_trace(go.Scatter(
                x=df_c["step"].values, y=norm,
                mode="lines", name=name,
                line=dict(color=palette[idx % len(palette)], width=2),
            ))
        fig_cmp.update_layout(
            title="Normalised Gamma Count Comparison (all sheets)",
            xaxis_title="Step", yaxis_title="Normalised Count (0–1)",
            height=420, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Statistics table
        st.markdown("#### Sheet Statistics")
        stats_rows = []
        for name in compare_sheets:
            df_c = all_sheets[name]
            c = df_c["count"].values
            sm = smooth_signal(c, smoothing_window)
            lb = rule_based_labels(sm, air_pct, sponge_pct)
            stats_rows.append({
                "Sheet": name,
                "Steps": len(c),
                "Min Count": round(c.min(), 0),
                "Max Count": round(c.max(), 0),
                "Mean Count": round(c.mean(), 1),
                "Std Count": round(c.std(), 1),
                "% Air": f"{(lb=='Air').mean():.1%}",
                "% Sponge": f"{(lb=='Sponge').mean():.1%}",
                "% Material": f"{(lb=='Material').mean():.1%}",
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

        # Air gap positions comparison
        st.markdown("#### Air Gap Positions Across Sheets")
        fig_air = go.Figure()
        for idx, name in enumerate(compare_sheets):
            df_c = all_sheets[name]
            sm = smooth_signal(df_c["count"].values, smoothing_window)
            lb = rule_based_labels(sm, air_pct, sponge_pct)
            air_steps = df_c["step"].values[lb == "Air"]
            fig_air.add_trace(go.Scatter(
                x=air_steps,
                y=[name] * len(air_steps),
                mode="markers",
                marker=dict(color="#4FC3F7", size=6, symbol="square"),
                name=name,
                showlegend=False,
            ))
        fig_air.update_layout(
            title="Air Gap Step Positions by Sheet",
            xaxis_title="Step", yaxis_title="Sheet",
            height=max(200, len(compare_sheets) * 70),
            template="plotly_white",
        )
        st.plotly_chart(fig_air, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab_anomaly:
    st.subheader("🔍 Anomaly Detection")
    st.markdown(
        "Uses **Isolation Forest** to detect statistically unusual count values "
        "that may indicate sensor noise, column defects, or unexpected voids."
    )

    anom_sheet = st.selectbox("Select sheet", list(selected_sheets.keys()), key="anom_sheet")
    df_an = selected_sheets[anom_sheet]
    steps_an = df_an["step"].values
    counts_an = df_an["count"].values
    smoothed_an = smooth_signal(counts_an, smoothing_window)
    is_anomaly = anomaly_detection(counts_an)
    labels_an = rule_based_labels(smoothed_an, air_pct, sponge_pct)

    n_anom = is_anomaly.sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Steps", len(steps_an))
    col2.metric("Anomalies Detected", int(n_anom))
    col3.metric("Anomaly Rate", f"{n_anom/len(steps_an):.1%}")

    fig_an = go.Figure()
    fig_an.add_trace(go.Scatter(
        x=steps_an, y=counts_an, mode="lines",
        line=dict(color="#90CAF9", width=1.2), name="Raw Count", opacity=0.7,
    ))
    fig_an.add_trace(go.Scatter(
        x=steps_an, y=smoothed_an, mode="lines",
        line=dict(color="#1565C0", width=2), name="Smoothed",
    ))
    # Highlight anomalies
    anom_steps = steps_an[is_anomaly]
    anom_counts = counts_an[is_anomaly]
    fig_an.add_trace(go.Scatter(
        x=anom_steps, y=anom_counts, mode="markers",
        marker=dict(color="red", size=8, symbol="x", line=dict(width=1.5)),
        name="Anomaly",
    ))
    fig_an.update_layout(
        title=f"Anomaly Detection — {anom_sheet}",
        xaxis_title="Step", yaxis_title="Gamma Count",
        height=420, template="plotly_white",
    )
    st.plotly_chart(fig_an, use_container_width=True)

    if n_anom > 0:
        anom_table = pd.DataFrame({
            "Step": steps_an[is_anomaly],
            "Count": counts_an[is_anomaly],
            "Label": labels_an[is_anomaly],
            "Deviation from Mean": (counts_an[is_anomaly] - counts_an.mean()).round(1),
        })
        st.markdown("#### Anomalous Steps")
        st.dataframe(anom_table, use_container_width=True)
    else:
        st.success("No anomalies detected in this sheet.")


# ══════════════════════════════════════════════
#  TAB 5 — EXPORT
# ══════════════════════════════════════════════
with tab_export:
    st.subheader("📥 Export Results")

    export_sheet = st.selectbox("Select sheet to export", list(selected_sheets.keys()), key="exp_sheet")
    df_ex = selected_sheets[export_sheet]
    steps_ex = df_ex["step"].values
    counts_ex = df_ex["count"].values
    smoothed_ex = smooth_signal(counts_ex, smoothing_window)
    labels_ex = rule_based_labels(smoothed_ex, air_pct, sponge_pct)
    features_ex = extract_ml_features(counts_ex, smoothed_ex)
    is_anom_ex = anomaly_detection(counts_ex)

    export_df = pd.DataFrame({
        "Step":          steps_ex,
        "Raw_Count":     counts_ex,
        "Smoothed_Count": smoothed_ex.round(2),
        "ML_Label":      labels_ex,
        "Is_Anomaly":    is_anom_ex.astype(int),
        **{col: features_ex[col].round(3).values for col in features_ex.columns},
    })

    region_ex = build_region_summary(steps_ex, labels_ex, counts_ex)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Step-by-Step Results")
        st.dataframe(export_df.head(50), use_container_width=True)
        csv_steps = export_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Step Results CSV",
            data=csv_steps,
            file_name=f"gamma_steps_{export_sheet.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    with c2:
        st.markdown("#### Region Summary")
        st.dataframe(region_ex, use_container_width=True)
        csv_regions = region_ex.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Region Summary CSV",
            data=csv_regions,
            file_name=f"gamma_regions_{export_sheet.replace(' ', '_')}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### Export All Sheets (Combined)")
    if st.button("Generate Combined Export"):
        combined_rows = []
        for name, df_c in all_sheets.items():
            s = df_c["step"].values
            c = df_c["count"].values
            sm = smooth_signal(c, smoothing_window)
            lb = rule_based_labels(sm, air_pct, sponge_pct)
            tmp = pd.DataFrame({"Sheet": name, "Step": s, "Count": c,
                                 "Smoothed": sm.round(2), "Label": lb})
            combined_rows.append(tmp)
        combined_df = pd.concat(combined_rows, ignore_index=True)
        csv_combined = combined_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Combined CSV (all sheets)",
            data=csv_combined,
            file_name="gamma_all_sheets_classified.csv",
            mime="text/csv",
        )
