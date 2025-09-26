import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import requests
import zipfile

# --- PATHS ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VISUALS_DIR = REPO_ROOT / "visuals"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline_rf_v1.pkl"
DATASET_ZIP = REPO_ROOT / "dataset.zip"

# --- GITHUB RELEASE URL ---
GITHUB_RELEASE_URL = "https://github.com/kimenjuivy/Netflix-Churn-Data-Analysis/releases/download/v1.0/dataset.zip"

# --- DATASET HANDLING ---
def ensure_dataset_extracted():
    """Ensure raw dataset exists locally; if missing, download from GitHub Releases."""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        if not DATASET_ZIP.exists():
            r = requests.get(GITHUB_RELEASE_URL)
            r.raise_for_status()
            with open(DATASET_ZIP, "wb") as f:
                f.write(r.content)
        # extract zip
        with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
        return True, "Dataset ready"
    except Exception as e:
        return False, f"Failed to prepare dataset: {e}"

def load_raw_telco():
    """Load raw churn dataset from raw folder."""
    try:
        # Assuming CSV name inside zip: cleaned_churn_data.csv
        csv_file = RAW_DIR / "cleaned_churn_data.csv"
        if not csv_file.exists():
            ensure_dataset_extracted()
        df = pd.read_csv(csv_file)
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {e}"

# --- MODEL HANDLING ---
def load_model():
    """Load trained model from models folder."""
    try:
        if not MODEL_PATH.exists():
            return None, f"{MODEL_PATH.name} not found in models/"
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

def get_model_input_features(model):
    """Attempt to infer input features from the model (if pipeline)."""
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        return None
    except Exception:
        return None

# --- VISUALS ---
def list_visuals():
    """Return a sorted list of all visuals in the visuals folder (PNG/JPG/HTML)."""
    if not VISUALS_DIR.exists():
        VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(VISUALS_DIR.glob("*"))
    return [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".html"]]

# --- JSON HANDLING FOR SINGLE-PREDICTION ---
def safe_parse_json_row(json_str):
    """
    Safely parse a single-row or list-of-rows JSON string into a DataFrame.
    Returns (DataFrame, error message)
    """
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = [data]  # wrap single row into list
        df = pd.DataFrame(data)
        return df, None
    except Exception as e:
        return None, f"Invalid JSON: {e}"

# --- DOWNLOAD LINK FOR PREDICTIONS ---
def df_to_download_link(df, filename="predictions.csv"):
    """Return a HTML link to download a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

# --- MISC ---
import base64
