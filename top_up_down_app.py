import streamlit as st
import pandas as pd
import pickle
import re

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

# ── Session state ────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Metrics banner ───────────────────────────────────────────────────────────
st.markdown("#### Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
c2.metric("Accuracy", f"{metrics['acc_mean']:.3f}")
c3.metric("F1",       f"{metrics['f1_mean']:.3f}")
st.divider()

# ── Helper: normalise miRNA name for fuzzy matching ──────────────────────────
def normalize(name: str) -> str:
    """Strip 5p/3p suffix and lowercase for loose matching."""
    return re.sub(r'-(5p|3p)$', '', name.strip().lower())

# ── Helper: resolve a user-typed miRNA to its lookup entry ──────────────────
def resolve_mirna(user_input: str):
    """
    Try three resolution strategies in order:
      1. Exact accession match  (MIMAT0000062)
      2. Exact miRNA name match (hsa-miR-21-5p)
      3. Fuzzy match after stripping arm suffix
    Returns (group_simplified, family_name, accession, None) or None.
    """
    user_input = user_input.strip()
    if user_input in accession_lookup:
        e = accession_lookup[user_input]
        return e['microrna_group_simplified'], e['family_name'], user_input, None
    if user_input in mirna_lookup:
        e = mirna_lookup[user_input]
        return e['microrna_group_simplified'], e['family_name'], e.get('mirbase_accession'), None
    norm_input = normalize(user_input)
    for key, val in mirna_lookup.items():
        if normalize(key) == norm_input:
            return val['microrna_group_simplified'], val['family_name'], val.get('mirbase_accession'), None
    return None

# ── Helper: build one prediction input row ──────────────────────────────────
def build_input_row(family_name_val, parasite, organism, cell_type, time_val):
    """
    Constructs the DataFrame row the model expects.
    family_name_val is None when the miRNA has no known family (not_found).
    In that case is_conserved=0 and family_name=None so the model falls back
    entirely on organism, parasite, cell type, and time.
    """
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
# TAB LAYOUT
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🔬 Single miRNA Prediction", "📊 Top miRNA Rankings"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — original single-miRNA predictor (unchanged logic)
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Prediction Inputs")

    mirna_input = st.text_input("miRNA name", placeholder="e.g. hsa-miR-21, miR-155-5p")
    parasite    = st.selectbox("Parasite",
                               ["L.major", "L.donovani", "L.amazonensis", "L. donovani"])
    organism    = st.selectbox("Organism", ["Human", "Mouse"])
    cell_type   = st.selectbox("Cell type",
                               ["PBMC", "THP-1", "BMDM (BALB/c females)",
                                "RAW 264.7", "Blood serum + liver (BALB/c )"])
    time        = st.number_input("Time (hours post-infection)", min_value=1, value=24)

    resolved = None
    if mirna_input:
        resolved = resolve_mirna(mirna_input)
        if resolved:
            group, family, accession, _ = resolved
            fam_display = family if (family and family not in ('unknown_family', 'not_found')) \
                          else 'Not conserved'
            st.success(f"**miRNA group:** `{group}`")
            col1, col2 = st.columns(2)
            col1.metric("Family name", fam_display)
            col2.metric("Accession",   accession or "N/A")
        else:
            st.warning("miRNA not found in lookup. Prediction will use unknown family.")

    if st.button("Predict", type="primary"):
        if not mirna_input.strip():
            st.warning("Please enter a miRNA name.")
        else:
            if resolved:
                group, family, _, _ = resolved
            else:
                group  = re.sub(r'^[a-z]{3}-', '', mirna_input.strip().lower())
                group  = re.sub(r'-(5p|3p)$', '', group)
                family = None

            input_df = build_input_row(family, parasite, organism, cell_type, time)
            proba    = model.predict_proba(input_df)[0][1]
            pred     = int(proba >= 0.5)
            label    = "⬆️ Upregulated" if pred == 1 else "⬇️ Downregulated"
            color    = "green" if pred == 1 else "red"

            st.markdown(f"### Prediction: :{color}[{label}]")
            st.metric("Confidence", f"{proba * 100:.1f}%")

            fam_display = family if (family and family not in ('unknown_family', 'not_found')) \
                          else 'Not conserved'
            st.session_state.history.append({
                "miRNA":      mirna_input.strip(),
                "Family":     fam_display,
                "Parasite":   parasite,
                "Organism":   organism,
                "Cell type":  cell_type,
                "Time (h)":   time,
                "Prediction": label,
                "Confidence": f"{proba * 100:.1f}%",
            })

    if st.session_state.history:
        st.subheader("Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Top miRNA Rankings (reverse lookup)
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Top miRNAs Predicted Up or Down Under Your Conditions")
    st.caption(
        "Enter your experimental conditions below. The model will score every "
        "miRNA in the database and return the most confidently up- and "
        "downregulated candidates."
    )

    # ── Condition inputs ─────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    rl_parasite = col_a.selectbox(
        "Parasite",
        ["L.major", "L.donovani", "L.amazonensis", "L. donovani"],
        key="rl_parasite"
    )
    rl_organism = col_b.selectbox(
        "Organism", ["Human", "Mouse"],
        key="rl_organism"
    )
    rl_cell = col_a.selectbox(
        "Cell type",
        ["PBMC", "THP-1", "BMDM (BALB/c females)",
         "RAW 264.7", "Blood serum + liver (BALB/c )"],
        key="rl_cell"
    )
    rl_time = col_b.number_input(
        "Time (hours post-infection)", min_value=1, value=24,
        key="rl_time"
    )
    rl_top_n = st.slider(
        "Number of miRNAs to show per direction", 5, 30, 10,
        key="rl_n"
    )

    if st.button("🔍 Rank All miRNAs", type="primary"):

        # ── Step 1: collect every unique miRNA from the lookup ───────────────
        # mirna_lookup is keyed by full miRNA name (e.g. "hsa-miR-21-5p").
        # We iterate over every entry so each miRNA is treated individually.
        # This means hsa-miR-9-5p and hsa-miR-9-3p get separate rows and
        # separate predictions even though they share a precursor — which is
        # the correct biological behaviour.

        rows = []
        for mirna_name, entry in mirna_lookup.items():
            family = entry.get('family_name', None)

            # Build the feature row for this specific miRNA.
            # build_input_row handles the not_found / unknown case:
            #   - if family is known  → family_name = family,  is_conserved = 1
            #   - if family unknown   → family_name = None,    is_conserved = 0
            #     the model then relies on organism, parasite, cell type, time
            input_df = build_input_row(
                family, rl_parasite, rl_organism, rl_cell, rl_time
            )

            # Get probability of being upregulated (class 1)
            proba = model.predict_proba(input_df)[0][1]

            # Decide direction based on 0.5 threshold
            direction = "⬆️ Up" if proba >= 0.5 else "⬇️ Down"

            # Display family cleanly
            fam_display = family if (
                family and family not in ('unknown_family', 'not_found')
            ) else "Unknown"

            rows.append({
                "miRNA":      mirna_name,
                "Family":     fam_display,
                "Direction":  direction,
                "P(up)":      proba,
                # Confidence = distance from 0.5, expressed as a clean %
                # e.g. P(up)=0.87 → confidence 87%, P(up)=0.13 → confidence 87%
                # This way both up and down confidences are comparable
                "Confidence": round(max(proba, 1 - proba) * 100, 1),
            })

        df_all = pd.DataFrame(rows)

        # ── Step 2: split and rank ───────────────────────────────────────────
        # Top upregulated   → highest P(up) first
        # Top downregulated → lowest  P(up) first (most confidently down)

        top_up = (
            df_all[df_all["P(up)"] >= 0.5]
            .sort_values("P(up)", ascending=False)
            .head(rl_top_n)
            .reset_index(drop=True)
        )
        top_up.index += 1  # rank starts at 1

        top_down = (
            df_all[df_all["P(up)"] < 0.5]
            .sort_values("P(up)", ascending=True)
            .head(rl_top_n)
            .reset_index(drop=True)
        )
        top_down.index += 1

        # ── Step 3: display ──────────────────────────────────────────────────
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
