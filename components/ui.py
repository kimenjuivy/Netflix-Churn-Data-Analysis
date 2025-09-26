# components/ui.py
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def section_header(title, subtitle=None, icon=""):
    if icon:
        st.markdown(f"### {icon} {title}")
    else:
        st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"_{subtitle}_")

def show_metrics_row(metrics: dict):
    """
    metrics: dict of label->value (or dict of dicts with 'value'/'delta')
    Example: {"AUC": "0.89", "Precision": "0.70"}
    """
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        if isinstance(v, dict):
            c.metric(k, v.get("value", ""), v.get("delta", ""))
        else:
            c.metric(k, v)

def display_image(path, caption=None, use_full_width=True):
    try:
        st.image(str(path), caption=caption or Path(path).name, use_container_width=use_full_width)
    except Exception as e:
        st.error(f"Unable to display image {path}: {e}")

def dataframe_preview(df, max_rows=200):
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head(max_rows))
    with st.expander("Show column types"):
        dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
        st.write(dtypes)
