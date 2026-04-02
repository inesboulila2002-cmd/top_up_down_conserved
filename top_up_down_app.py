import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Model206 conserved", page_icon="🧬", layout="wide")
st.title("🧬 miRNA Upregulation Predictor — Model 206 Accession")
st.caption("LightGBM · Target Encoding · conservation=2 · 5-fold CV")

# ── Load model bundle ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('Model206_conserved_model.pkl', 'rb') as f:
        return pickle.load(f)

bundle           = load_model()
model            = bundle['model']
mirna_lookup     = bundle['mirna_lookup']
accession_lookup = bundle['accession_lookup']
options          = bundle['options']
metrics          = bundle['metrics']

# ── Metrics banner ───────────────────────────────────────────────────────────
st.markdown("#### Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
c2.metric("Accuracy", f"{metrics['acc_mean']:.3f}")
c3.metric("F1",       f"{metrics['f1_mean']:.3f}")
st.divider()

# ── Helper: build one prediction input row ──────────────────────────────────
def build_input_row(family_name_val, parasite, organism, cell_type, time_val):
    fam_val = None if (
        not family_name_val or
        family_name_val in ('unknown_family', 'not_found')
    ) else family_name_val

    return pd.DataFrame([{
        'parasite':          parasite,
        'organism':          organism,
        'cell type':         cell_type,
        'family_name':       fam_val,
        'parasite_celltype': f"{parasite.strip()}_{cell_type.strip()}",
        'time':              int(time_val),
        'is_conserved':      0 if fam_val is None else 1,
    }])

# ════════════════════════════════════════════════════════════════════════════
# MAIN — Top miRNA Rankings ONLY
# ════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Top miRNAs Predicted Up or Down Under Your Conditions")
st.caption(
    "Enter your experimental conditions below. The model will score every "
    "miRNA in the database and return the most confidently up- and "
    "downregulated candidates."
)

# ── Condition inputs ─────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
rl_parasite = col_a.selectbox(
    "Parasite",
    ["L.major", "L.donovani", "L.amazonensis", "L. donovani"]
)
rl_organism = col_b.selectbox(
    "Organism", ["Human", "Mouse"]
)
rl_cell = col_a.selectbox(
    "Cell type",
    ["PBMC", "THP-1", "BMDM (BALB/c females)",
     "RAW 264.7", "Blood serum + liver (BALB/c )"]
)
rl_time = col_b.number_input(
    "Time (hours post-infection)", min_value=1, value=24
)
rl_top_n = st.slider(
    "Number of miRNAs to show per direction", 5, 30, 10
)

if st.button("🔍 Rank All miRNAs", type="primary"):

    rows = []
    for mirna_name, entry in mirna_lookup.items():
        family = entry.get('family_name', None)

        input_df = build_input_row(
            family, rl_parasite, rl_organism, rl_cell, rl_time
        )

        proba = model.predict_proba(input_df)[0][1]
        direction = "⬆️ Up" if proba >= 0.5 else "⬇️ Down"

        fam_display = family if (
            family and family not in ('unknown_family', 'not_found')
        ) else "Unknown"

        rows.append({
            "miRNA":      mirna_name,
            "Family":     fam_display,
            "Direction":  direction,
            "P(up)":      proba,
            "Confidence": round(max(proba, 1 - proba) * 100, 1),
        })

    df_all = pd.DataFrame(rows)

    # ── Ranking ───────────────────────────────────────────
    top_up = (
        df_all[df_all["P(up)"] >= 0.5]
        .sort_values("P(up)", ascending=False)
        .head(rl_top_n)
        .reset_index(drop=True)
    )
    top_up.index += 1

    top_down = (
        df_all[df_all["P(up)"] < 0.5]
        .sort_values("P(up)", ascending=True)
        .head(rl_top_n)
        .reset_index(drop=True)
    )
    top_down.index += 1

    # ── Display ──────────────────────────────────────────
    st.markdown(f"**Conditions:** {rl_parasite} · {rl_organism} · "
                f"{rl_cell} · {rl_time}h post-infection")
    st.markdown(f"Scored **{len(df_all)} miRNAs** — showing top {rl_top_n} per direction.")
    st.divider()

    col_up, col_down = st.columns(2)

    with col_up:
        st.markdown("### ⬆️ Top Upregulated")
        if top_up.empty:
            st.info("No miRNAs predicted as upregulated.")
        else:
            st.dataframe(
                top_up[["miRNA", "Family", "Confidence"]]
                .rename(columns={"Confidence": "Confidence (%)"}),
                use_container_width=True
            )

    with col_down:
        st.markdown("### ⬇️ Top Downregulated")
        if top_down.empty:
            st.info("No miRNAs predicted as downregulated.")
        else:
            st.dataframe(
                top_down[["miRNA", "Family", "Confidence"]]
                .rename(columns={"Confidence": "Confidence (%)"}),
                use_container_width=True
            )

    st.divider()
    st.warning(
        "⚠️ This model was trained on 206 samples. Rankings reflect learned "
        "patterns but should be validated experimentally."
    )
