import streamlit as st
import pandas as pd
import pickle

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="miRNA Expression Predictor",
    page_icon="🧬",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# Load model bundle (cached so it only loads once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle(path="top_up_down_conserved.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        "❌ Model file `Model206_conserved_model.pkl` not found. "
        "Please run the training notebook first to generate it, "
        "then place it in the same folder as this app."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────────────────────
def predict_top_mirnas(bundle, parasite, organism, cell_type, time, top_n=10):
    model    = bundle['model']
    families = bundle['options']['seed_families']
    detail   = bundle['family_detail_lookup']

    parasite = parasite.replace(' ', '')

    rows = [{
        'parasite':          parasite,
        'organism':          organism,
        'cell type':         cell_type,
        'time':              time,
        'family_name':       fam,
        'is_conserved':      1,
        'parasite_celltype': f"{parasite}_{cell_type}"
    } for fam in families]

    df_pred = pd.DataFrame(rows)
    df_pred['p_up'] = model.predict_proba(df_pred)[:, 1]

    records = []
    for _, row in df_pred.iterrows():
        fam     = row['family_name']
        members = detail.get(fam, [])

        mirna_names = ', '.join(m['microrna']  for m in members) if members else '—'
        accessions  = ', '.join(m['accession'] for m in members) if members else '—'
        groups      = ', '.join(dict.fromkeys(m['group'] for m in members)) if members else '—'

        records.append({
            'Seed Family':       fam,
            'miRNA Names':       mirna_names,
            'Accession Numbers': accessions,
            'miRNA Group':       groups,
            'P(upregulated)':    round(row['p_up'], 4),
            'P(downregulated)':  round(1 - row['p_up'], 4),
        })

    df_result = pd.DataFrame(records)

    top_up   = df_result.nlargest(top_n,  'P(upregulated)') \
                         .drop(columns='P(downregulated)') \
                         .reset_index(drop=True)
    top_up.index += 1

    top_down = df_result.nsmallest(top_n, 'P(upregulated)') \
                         .drop(columns='P(upregulated)') \
                         .reset_index(drop=True)
    top_down.index += 1

    return top_up, top_down


# ─────────────────────────────────────────────────────────────
# UI — Header
# ─────────────────────────────────────────────────────────────
st.title("🧬 miRNA Expression Predictor")
st.markdown(
    "Predict which miRNA families are most likely to be **upregulated** or "
    "**downregulated** given a parasite infection context."
)
st.divider()

# ─────────────────────────────────────────────────────────────
# UI — Sidebar inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔬 Query Parameters")
    st.markdown("Select your experimental conditions below.")

    options = bundle['options']

    parasite = st.selectbox(
        "Parasite",
        options=options['parasite'],
        help="The infecting parasite species"
    )

    organism = st.selectbox(
        "Organism (Host)",
        options=options['organism'],
        help="The host organism"
    )

    cell_type = st.selectbox(
        "Cell Type",
        options=options['cell_type'],
        help="The cell type or tissue"
    )

    time = st.selectbox(
        "Time Point (hours)",
        options=options['time'],
        help="Hours post-infection"
    )

    top_n = st.slider(
        "Number of top miRNAs to show",
        min_value=5,
        max_value=len(options['seed_families']),
        value=10,
        step=1
    )

    st.divider()
    run_button = st.button("🔍 Predict", use_container_width=True, type="primary")

    st.divider()
    st.caption(
        f"Model trained on **{bundle['metrics']['n_train']} samples** · "
        f"ROC-AUC **{bundle['metrics']['auc_mean']:.3f} ± {bundle['metrics']['auc_std']:.3f}**"
    )

# ─────────────────────────────────────────────────────────────
# UI — Main panel (results)
# ─────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Running prediction across all miRNA families..."):
        top_up, top_down = predict_top_mirnas(
            bundle    = bundle,
            parasite  = parasite,
            organism  = organism,
            cell_type = cell_type,
            time      = time,
            top_n     = top_n
        )

    # ── Summary banner ─────────────────────────────────────
    st.subheader("📋 Query Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parasite",   parasite)
    c2.metric("Organism",   organism)
    c3.metric("Cell Type",  cell_type)
    c4.metric("Time (hrs)", time)
    st.divider()

    # ── Results in two tabs ────────────────────────────────
    tab_up, tab_down = st.tabs([
        f"⬆️ Top {top_n} Upregulated",
        f"⬇️ Top {top_n} Downregulated"
    ])

    with tab_up:
        st.markdown(
            f"miRNA families **most likely to be upregulated** under these conditions, "
            f"ranked by predicted probability."
        )
        st.dataframe(
            top_up.style.background_gradient(
                subset=['P(upregulated)'], cmap='Greens'
            ).format({'P(upregulated)': '{:.4f}'}),
            use_container_width=True,
            height=min(50 + top_n * 38, 600)
        )

        csv_up = top_up.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="⬇️ Download upregulated results as CSV",
            data=csv_up,
            file_name=f"top_up_{parasite}_{cell_type}_{time}h.csv",
            mime="text/csv"
        )

    with tab_down:
        st.markdown(
            f"miRNA families **most likely to be downregulated** under these conditions, "
            f"ranked by predicted probability."
        )
        st.dataframe(
            top_down.style.background_gradient(
                subset=['P(downregulated)'], cmap='Reds'
            ).format({'P(downregulated)': '{:.4f}'}),
            use_container_width=True,
            height=min(50 + top_n * 38, 600)
        )

        csv_down = top_down.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="⬇️ Download downregulated results as CSV",
            data=csv_down,
            file_name=f"top_down_{parasite}_{cell_type}_{time}h.csv",
            mime="text/csv"
        )

    # ── Model metrics expander ─────────────────────────────
    with st.expander("📊 Model Performance Details"):
        m = bundle['metrics']
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ROC-AUC",  f"{m['auc_mean']:.3f} ± {m['auc_std']:.3f}")
        mc2.metric("Accuracy", f"{m['acc_mean']:.3f} ± {m['acc_std']:.3f}")
        mc3.metric("F1 Score", f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}")

        st.markdown("**Feature Importance (Permutation)**")
        fi_df = pd.DataFrame(m['feature_importance'])
        st.dataframe(fi_df.style.background_gradient(subset=['importance'], cmap='Blues'),
                     use_container_width=True)

else:
    # ── Placeholder before first run ──────────────────────
    st.info(
        "👈 Set your experimental conditions in the sidebar and click **Predict** "
        "to see the ranked miRNA list."
    )

    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**1. Select conditions**\n\n"
            "Choose your parasite, host organism, cell type, and time point."
        )
    with col2:
        st.markdown(
            "**2. Model scores all families**\n\n"
            "The LightGBM model scores every known miRNA seed family against your query."
        )
    with col3:
        st.markdown(
            "**3. Ranked results**\n\n"
            "Results show seed family, miRNA names, accession numbers, and predicted probability."
        )
