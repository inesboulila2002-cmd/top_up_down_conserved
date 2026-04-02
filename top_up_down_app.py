import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="miRNA Expression Predictor", page_icon="🧬", layout="wide")

@st.cache_resource
def load_bundle(path="top_up_down_conserved.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("❌ Model file `top_up_down_conserved.pkl` not found.")
    st.stop()

def predict_top_mirnas(bundle, parasite, organism, cell_type, time, top_n=10):
    model       = bundle['model']
    mirna_names = bundle['options']['mirna_names']
    fam_lookup  = bundle['mirna_family_lookup']

    parasite_clean = parasite.replace(' ', '')
    rows = []
    
    for mirna in mirna_names:
        # Get family name. Default to np.nan if not found.
        fam = fam_lookup.get(mirna, np.nan)
        
        # If family is NaN, is_conserved is 0. Otherwise 1.
        is_cons = 0 if pd.isna(fam) else 1
        
        rows.append({
            'parasite':          parasite_clean,
            'organism':          organism,
            'cell type':         cell_type,
            'time':              time,
            'microrna':          mirna,
            'family_name':       fam,  # <--- Passes pure NaN to the model
            'is_conserved':      is_cons,
            'parasite_celltype': f"{parasite_clean}_{cell_type}"
        })

    df_pred = pd.DataFrame(rows)
    probs = model.predict_proba(df_pred)[:, 1]

    df_result = pd.DataFrame({
        'miRNA Name':         mirna_names,
        'P(upregulated) %':   (probs * 100).round(2),
        'P(downregulated) %': ((1 - probs) * 100).round(2),
    })

    top_up = df_result.nlargest(top_n, 'P(upregulated) %').reset_index(drop=True)
    top_down = df_result.nlargest(top_n, 'P(downregulated) %').reset_index(drop=True)
    top_up.index += 1
    top_down.index += 1

    return top_up, top_down

# UI Header
st.title("🧬 miRNA Expression Predictor")

with st.sidebar:
    st.header("🔬 Query Parameters")
    opt = bundle['options']
    parasite = st.selectbox("Parasite species", opt['parasite'])
    organism = st.selectbox("Host organism", opt['organism'])
    cell_type = st.selectbox("Cell type", opt['cell_type'])
    time = st.selectbox("Time (hours)", opt['time'])
    top_n = st.slider("Results count", 5, len(opt['mirna_names']), 10)
    run_button = st.button("🔍 Predict", use_container_width=True, type="primary")

if run_button:
    top_up, top_down = predict_top_mirnas(bundle, parasite, organism, cell_type, time, top_n)
    
    st.subheader("📋 Query Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parasite", parasite); c2.metric("Organism", organism); 
    c3.metric("Cell Type", cell_type); c4.metric("Time", f"{time}h")

    t_up, t_down = st.tabs(["⬆️ Top Upregulated", "⬇️ Top Downregulated"])
    with t_up: st.dataframe(top_up.style.background_gradient(subset=['P(upregulated) %'], cmap='Greens'), use_container_width=True)
    with t_down: st.dataframe(top_down.style.background_gradient(subset=['P(downregulated) %'], cmap='Reds'), use_container_width=True)

    with st.expander("📊 Model Performance Details"):
        m = bundle['metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC", f"{m['auc_mean']:.3f}")
        col2.metric("Accuracy", f"{m['acc_mean']:.3f}")
        col3.metric("F1 Score", f"{m['f1_mean']:.3f}")
        st.write("**Feature Importance**")
        st.table(pd.DataFrame(m['feature_importance']))
else:
    st.info(" Select experimental conditions to begin.")