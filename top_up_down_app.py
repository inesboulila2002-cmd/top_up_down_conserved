import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🧬 miRNA Predictor — Dual Model")

mode = st.radio("Choose model type", ["No Family (Safer)", "With Family (Biological)"])

file = "model_no_family.pkl" if "No Family" in mode else "model_with_family.pkl"

bundle = pickle.load(open(file, "rb"))
model = bundle['model']
opt   = bundle['options']

parasite = st.selectbox("Parasite", opt['parasite'])
organism = st.selectbox("Organism", opt['organism'])
cell_type = st.selectbox("Cell type", opt['cell_type'])
time = st.selectbox("Time", opt['time'])

if st.button("Predict Top miRNAs"):

    rows = []
    for mirna in opt['mirna_names']:
        rows.append({
            'parasite': parasite,
            'organism': organism,
            'cell type': cell_type,
            'time': time,
            'microrna': mirna,
            'family_name': np.nan,  # ignored in no-family model
            'is_conserved': 1,
            'parasite_celltype': f"{parasite}_{cell_type}"
        })

    df = pd.DataFrame(rows)
    probs = model.predict_proba(df)[:,1]

    res = pd.DataFrame({
        'miRNA': opt['mirna_names'],
        'Up %': (probs*100).round(2)
    }).sort_values('Up %', ascending=False)

    st.dataframe(res.head(10))

    st.subheader("Feature Importance")
    st.dataframe(pd.DataFrame(bundle['importance']))