# components/utils.py
from pathlib import Path
import zipfile
import joblib
import pandas as pd
import base64
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ZIP = REPO_ROOT / "dataset.zip"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
MODELS_DIR = REPO_ROOT / "models"
VISUALS_DIR = REPO_ROOT / "visuals"
MODEL_PATH = MODELS_DIR / "churn_pipeline_rf_v1.pkl"

# ensure folders exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

def ensure_dataset_extracted():
    """
    Extract dataset.zip into data/raw/ if not already present.
    Returns tuple (ok:bool, message:str)
    """
    # if data/raw is non-empty, assume already extracted
    try:
        if any(DATA_RAW_DIR.iterdir()):
            return True, f"Data already present under {DATA_RAW_DIR}"
    except FileNotFoundError:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_ZIP.exists():
        try:
            with zipfile.ZipFile(DATA_ZIP, "r") as z:
                z.extractall(DATA_RAW_DIR)
            return True, f"Extracted {DATA_ZIP.name} -> {DATA_RAW_DIR}"
        except Exception as e:
            return False, f"Failed to extract {DATA_ZIP}: {e}"
    return False, f"dataset.zip not found at {DATA_ZIP}. Please place dataset.zip in repo root."

def find_raw_csv(preferred_contains="WA_Fn-UseC_-Telco-Customer-Churn"):
    """
    Return Path to a raw CSV (prefer telco churn file), or None
    """
    csvs = sorted(DATA_RAW_DIR.glob("*.csv"))
    if not csvs:
        return None
    for c in csvs:
        if preferred_contains in c.name:
            return c
    return csvs[0]

def load_raw_telco(preferred_contains="WA_Fn-UseC_-Telco-Customer-Churn"):
    """
    Load telco CSV after extraction. Returns (df, error)
    """
    ok, msg = ensure_dataset_extracted()
    if not ok and not any(DATA_RAW_DIR.iterdir()):
        return None, msg
    csv_path = find_raw_csv(preferred_contains)
    if not csv_path:
        return None, f"No CSV found in {DATA_RAW_DIR}"
    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Failed to read {csv_path}: {e}"

def load_model(path: Path = None):
    """
    Load joblib model. Returns (model, error)
    """
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        return None, f"Model not found at {p}. Place churn_pipeline_rf_v1.pkl into models/."
    try:
        m = joblib.load(p)
        return m, None
    except Exception as e:
        return None, f"Failed to load model at {p}: {e}"

def get_model_input_features(model):
    """
    Try to get model input feature names. Returns list or None.
    Compatible with sklearn pipeline having preprocessor or feature_names_in_
    """
    if model is None:
        return None
    # direct attribute
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass
    # pipeline preprocessor
    try:
        if hasattr(model, "named_steps"):
            pre = model.named_steps.get("preprocessor")
            if pre is not None:
                # try get_feature_names_out (safe)
                try:
                    return list(pre.get_feature_names_out())
                except Exception:
                    # fallback to recorded names if available
                    if hasattr(pre, "feature_names_in_"):
                        return list(pre.feature_names_in_)
    except Exception:
        pass
    # classifier fallback
    try:
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            clf = model.named_steps["classifier"]
            if hasattr(clf, "feature_names_in_"):
                return list(clf.feature_names_in_)
    except Exception:
        pass
    return None

def list_visuals(extensions=(".png", ".jpg", ".jpeg", ".html")):
    """List files in visuals/ with specified extensions."""
    files = [p for p in sorted(VISUALS_DIR.iterdir()) if p.suffix.lower() in extensions]
    return files

def df_to_download_link(df, filename="download.csv"):
    """Return HTML anchor tag to download a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def safe_parse_json_row(text):
    """Parse a single-row JSON input, return DataFrame or (None,error)"""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return pd.DataFrame([obj]), None
        elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            return pd.DataFrame(obj), None
        else:
            return None, "JSON must be a dict or list-of-dicts representing one or more rows."
    except Exception as e:
        return None, f"Invalid JSON: {e}"
