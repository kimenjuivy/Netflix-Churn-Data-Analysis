import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import traceback
import streamlit.components.v1 as components

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Components
from components import utils, ui

# Config
st.set_page_config(
    page_title="Netflix Churn — Executive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

VISUALS_DIR = REPO_ROOT / "visuals"

# Sidebar
st.sidebar.title("📊 Netflix Churn")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio("", ["Home", "Data Explorer", "Model Playground", "Insights", "Business Impact", "Deploy"])

# Status checks
ok, msg = utils.ensure_dataset_extracted()
st.sidebar.success("Data ready" if ok else msg)

model_obj, model_err = utils.load_model()
st.sidebar.success("Model loaded" if model_obj else model_err)
model_features = utils.get_model_input_features(model_obj)

# --- HOME ---
if page == "Home":
    st.title("🎬 Netflix Churn — Executive Dashboard")
    st.markdown("""
    This interactive app presents the churn prediction model, validation metrics, 
    and business impact analyses produced in the Netflix Churn Analysis project.
    """)
    st.markdown("**Quick links:** - Data Explorer  •  Model Playground  •  Business Impact")

    # show top visual if present
    top_visual = utils.list_visuals()
    if top_visual:
        for v in top_visual[:1]:  # show first visual only
            if v.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                ui.display_image(v, caption=v.name)
            elif v.suffix.lower() == ".html":
                with open(v, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=500, scrolling=True)
    else:
        st.info("No visuals found in visuals/. Copy generated images to visuals/")

# --- DATA EXPLORER ---
elif page == "Data Explorer":
    st.header("Data Explorer")
    df_raw, err = utils.load_raw_telco()
    if err:
        st.error(err)
    else:
        ui.section_header("Raw dataset preview", f"{df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        ui.dataframe_preview(df_raw)

    st.markdown("---")
    st.subheader("Saved visuals gallery")
    files = utils.list_visuals()
    if files:
        cols = st.columns(3)
        for i, f in enumerate(files):
            with cols[i % 3]:
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    ui.display_image(f, caption=f.name)
                elif f.suffix.lower() == ".html":
                    with open(f, "r", encoding="utf-8") as hf:
                        components.html(hf.read(), height=400, scrolling=True)
    else:
        st.info("No visuals present. Copy PNGs/HTMLs into visuals/ folder.")

# --- MODEL PLAYGROUND ---
elif page == "Model Playground":
    st.header("Model Playground")
    if model_obj is None:
        st.error("Model not available. Put churn_pipeline_rf_v1.pkl into models/ and reload.")
        st.stop()

    st.subheader("Model summary")
    st.write(type(model_obj))
    if model_features:
        with st.expander("Expected input columns (inferred)"):
            st.write(model_features)
    else:
        st.warning("Could not infer expected feature names from pipeline.")

    # Single-row JSON input
    st.subheader("Single-customer prediction (JSON)")
    sample_json = '{"tenure_months": 12, "monthly_cost": 9.99, "contract_type": "Month-to-month", "engagement_level": "Low"}'
    user_json = st.text_area("Paste single-row JSON or list-of-rows JSON:", value=sample_json, height=120)
    if st.button("Predict from JSON"):
        try:
            df_in, err = utils.safe_parse_json_row(user_json)
            if err:
                st.error(err)
            else:
                missing = [c for c in model_features if c not in df_in.columns] if model_features else []
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    pred = model_obj.predict(df_in)
                    proba = model_obj.predict_proba(df_in)[:, 1]
                    st.write("Prediction:", pred[0])
                    st.metric("Churn probability", f"{proba[0]:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.text(traceback.format_exc())

    st.markdown("---")
    st.subheader("Batch predictions (CSV)")
    uploaded = st.file_uploader("Upload cleaned CSV", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            st.write(f"Uploaded: {df_up.shape[0]} rows")
            missing = [c for c in model_features if c not in df_up.columns] if model_features else []
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                preds = model_obj.predict(df_up)
                probs = model_obj.predict_proba(df_up)[:, 1]
                out = df_up.copy()
                out["churn_pred"] = preds
                out["churn_proba"] = probs
                st.dataframe(out.head(50), use_container_width=True)
                st.markdown(utils.df_to_download_link(out, "predictions.csv"), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.text(traceback.format_exc())

# --- INSIGHTS ---
elif page == "Insights":
    st.header("Insights & Model Diagnostics")
    st.markdown("**Key saved visuals**")
    files = utils.list_visuals()
    if files:
        for p in files:
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                ui.display_image(p, caption=p.name)
            elif p.suffix.lower() == ".html":
                with open(p, "r", encoding="utf-8") as hf:
                    components.html(hf.read(), height=400, scrolling=True)
    else:
        st.info("No visuals found in visuals/")

    st.markdown("---")
    st.subheader("Saved CV AUC scores")
    cv_path = VISUALS_DIR / "cv_auc_scores.csv"
    st.table(pd.read_csv(cv_path)) if cv_path.exists() else st.info("cv_auc_scores.csv not found")

# --- BUSINESS IMPACT ---
elif page == "Business Impact":
    st.header("Business Impact & ROI")
    rcb = VISUALS_DIR / "retention_cost_benefit.csv"
    st.dataframe(pd.read_csv(rcb)) if rcb.exists() else st.info("retention_cost_benefit.csv not found")

    heat = VISUALS_DIR / "sensitivity_heatmap.png"
    if heat.exists():
        ui.display_image(heat, caption="Sensitivity Heatmap")
    else:
        st.info("sensitivity_heatmap.png not found")

# --- DEPLOY ---
elif page == "Deploy":
    st.header("Deploy / Notes")
    st.markdown("""
    1. Push all code to GitHub (app/, models/, visuals/, requirements.txt).
    2. Add dataset.zip to GitHub Releases for download.
    3. Connect the repo to Streamlit Cloud, set main file: app/app.py.
    """)
    st.markdown("**Sanity checks:**")
    st.write(f"- Model file: {'found' if (Path(utils.MODEL_PATH).exists()) else 'missing'}")
    st.write(f"- Visuals: {len(utils.list_visuals())} files found")
    st.write(f"- Extracted data: {'yes' if any((REPO_ROOT / 'data' / 'raw').glob('*.csv')) else 'no'}")
