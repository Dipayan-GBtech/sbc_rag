# streamlit_app.py
# Uses your aca_mod exactly as-is.
# - Normalizes input columns so aca_mod.adjudicate_row gets what it expects.
# - Submit adjudicates ALL rows for the claim (multi-line) by calling adjudicate_row per row.
# - Rule box shows Rule, Reason, Prompt, and parsed JSON from retrieve_hierarchically.

import re
import json
import pathlib
import pandas as pd
import streamlit as st

from aca_mod import adjudicate_row, adjudicate_claim_by_no, safe_parse_json

# -----------------------------
# Config & style
# -----------------------------
st.set_page_config(page_title="GBT-Claim", page_icon="🧾", layout="wide")

PRIMARY = "#0E8F57"
ACCENT = "#0fb56e"

st.markdown(f"""
<style>
.header {{
  background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
  color: white; border-radius: 16px; padding: 14px 18px; margin-bottom: 12px;
}}
.header .title {{ font-size: 24px; font-weight: 700; }}
.card {{
  background: #ffffff; border-radius: 16px; padding: 16px; border: 1px solid #ececec;
  box-shadow: 0 4px 12px rgba(0,0,0,0.04); margin-bottom: 10px;
}}
.label {{ color: #637381; font-size: 12px; }}
.value {{ color: #111; font-weight: 600; }}
.rule-box {{
  background: #f4fbf7; border: 1px dashed {PRIMARY}; border-radius: 12px; padding: 12px;
  font-size: 13px;
}}
.subtle {{ color:#6b7280; font-size: 13px; }}
.stButton>button.primary {{ background: {PRIMARY}; color: white; border: 0; border-radius: 10px; padding: 8px 16px; font-weight: 600; }}
.stButton>button.secondary {{ background: white; color: {PRIMARY}; border: 1px solid {PRIMARY}; border-radius: 10px; padding: 8px 16px; font-weight: 600; }}
.smallcaps {{ font-variant: all-small-caps; letter-spacing: .04em; color:#6b7280; }}
code {{ white-space: pre-wrap; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
DATA_DIR = pathlib.Path(".")
UPLOAD_DIR = pathlib.Path("sbc_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SETTLEMENT_FIELDS = [
    "PLAN_PAID",
    "PROVIDER_RESPONSIBILITY",
    "CO_PAY",
    "DEDUCTIBLE",
    "CO_INSURANCE",
    "EMPLOYEE_RESPONSIBILITY"
]
ALL_FIELDS = ["CHARGE_AMOUNT"] + SETTLEMENT_FIELDS

def _safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            pass

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_column_by_keywords(df: pd.DataFrame, must_have=None, any_of=None):
    must_have = [norm(x) for x in (must_have or [])]
    any_of = [norm(x) for x in (any_of or [])]
    candidates = []
    for c in df.columns:
        n = norm(c)
        if all(t in n for t in must_have) and (not any_of or any(t in n for t in any_of)):
            candidates.append(c)
    candidates.sort(key=lambda x: len(x), reverse=True)
    return candidates[0] if candidates else None

def find_claim_column(df: pd.DataFrame) -> str:
    for pat in ["claim_no", "claimnumber", "claim_num", "claimid", "claim", "claim number"]:
        for c in df.columns:
            if norm(pat) == norm(c):
                return c
    col = find_column_by_keywords(df, must_have=["claim"], any_of=["no","num","id","number"])
    return col or df.columns[0]

def find_charge_column(df: pd.DataFrame) -> str:
    for pat in ["charge_amount", "total_charge", "billed_amount", "claim_amount", "amount_billed"]:
        for c in df.columns:
            if norm(pat) == norm(c):
                return c
    col = find_column_by_keywords(df, must_have=["charge"], any_of=["amount","amt","total","billed"])
    if not col:
        col = find_column_by_keywords(df, must_have=["amount"], any_of=["charge","billed","total"])
    return col or df.columns[0]

def robust_get(row: pd.Series, synonyms: list, default="—"):
    idxmap = {norm(c): c for c in row.index}
    for s in synonyms:
        ns = norm(s)
        if ns in idxmap:
            v = row[idxmap[ns]]
            return default if (pd.isna(v) or str(v).strip() == "") else v
        for nk, orig in idxmap.items():
            if ns in nk:
                v = row[orig]
                return default if (pd.isna(v) or str(v).strip() == "") else v
    return default

def render_kv(col, label, value):
    col.markdown("<div class='label'>{}</div>".format(label), unsafe_allow_html=True)
    col.markdown("<div class='value'>{}</div>".format("—" if value in [None, "nan", "—"] or str(value).strip()=="" else value), unsafe_allow_html=True)

def load_claims_input() -> pd.DataFrame:
    for fname in ["train1_to.csv", "train_1to.csv", "train_to.csv"]:
        p = DATA_DIR / fname
        if p.exists():
            df = pd.read_csv(p)
            # normalize like aca_mod does
            df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_", regex=False).str.replace(".", "", regex=False)
            return df
    return pd.DataFrame()

def safe_float(x) -> float:
    try:
        if isinstance(x, str):
            import re as _re
            m = _re.search(r"-?\d+(\.\d+)?", x.replace(",", ""))
            return float(m.group()) if m else 0.0
        return float(x)
    except Exception:
        return 0.0

def to_ui_row(orig_row: pd.Series, adjudicated: dict, charge_col: str) -> dict:
    """Map aca_mod.adjudicate_row result to UI columns (+ raw fields for rule panel)."""
    charge_amt = safe_float(orig_row.get(charge_col, 0.0))
    rag_json = safe_parse_json(adjudicated.get("rag_response", "")) if adjudicated.get("rag_response") else {}
    return {
        # editor columns
        "CHARGE_AMOUNT": round(charge_amt, 2),
        "PLAN_PAID": adjudicated.get("plan_paid", 0.0),
        "PROVIDER_RESPONSIBILITY": adjudicated.get("provider_responsibility", 0.0),
        "CO_PAY": adjudicated.get("co_pay", 0.0),
        "DEDUCTIBLE": adjudicated.get("deductible", 0.0),
        "CO_INSURANCE": adjudicated.get("co_insurance", 0.0),
        "EMPLOYEE_RESPONSIBILITY": adjudicated.get("employee_responsibility", 0.0),
        # rule panel data
        "_rule": adjudicated.get("settlement_rule", "—"),
        "_reason": adjudicated.get("settlement_reason_log", "—"),
        "_prompt": adjudicated.get("rag_prompt", ""),
        "_raw": adjudicated.get("rag_response", ""),
        "_json": rag_json,
    }

# -----------------------------
# Header
# -----------------------------
c1, c2 = st.columns([3,1])
with c1:
    st.markdown("<div class='header'><div class='title'>🔎 Find Your Claims Today!</div></div>", unsafe_allow_html=True)

# -----------------------------
# Search Bar (+ optional SBC upload shell)
# -----------------------------
bar_left, bar_right = st.columns([4.5, 1.5])
with bar_left:
    search_claim = st.text_input("Search by Claim Number", value="", placeholder="0000-000000000-0000")
with bar_right:
    sbc_choice=st.selectbox("SBC Upload", options=["No", "Yes"], index=0, disabled=False)
    if sbc_choice == "Yes":
        with st.expander("Upload New SBC file", expanded=True):
            up = st.file_uploader("Choose SBC / Plan PDF or DOCX", type=["pdf", "docx", "txt"])
            if up is not None:
                path = UPLOAD_DIR / up.name
                with open(path, "wb") as f:
                    f.write(up.read())
                st.success(f"Uploaded to: {path}")

claims_df = load_claims_input()

if "show_journey_modal" not in st.session_state:
    st.session_state["show_journey_modal"] = False
if "adjudicated_rows" not in st.session_state:
    st.session_state["adjudicated_rows"] = []

# -----------------------------
# When a claim is searched
# -----------------------------
if search_claim.strip():
    if claims_df.empty:
        st.error("Input dataset not found. Ensure `train1_to.csv` (or `train_1to.csv`/`train_to.csv`) exists.")
        st.stop()

    claim_col_in = find_claim_column(claims_df)
    charge_col_in = find_charge_column(claims_df)

    rows = claims_df[claims_df[claim_col_in].astype(str).str.strip().str.lower()
                     == str(search_claim).strip().lower()]
    if rows.empty:
        st.warning("Claim number does not exist.")
        st.stop()

    # Claimant & Provider cards from first line
    first_row = rows.iloc[0]
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='card'><h4>Claimant Details</h4>", unsafe_allow_html=True)
        g1a, g1b, g1c, g1d = st.columns(4)
        render_kv(g1a, "Claim Number", str(first_row.get(claim_col_in, "—")))
        render_kv(g1b, "Employee Number", robust_get(first_row, ["employee_number","employee id","EMPNO","subscriber id"]))
        render_kv(g1c, "Patient Name", robust_get(first_row, ["patient_name","member name","subscriber name","name"]))
        render_kv(g1d, "Patient DOB", robust_get(first_row, ["patient_dob","dob","birth date"]))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>Provider Details</h4>", unsafe_allow_html=True)
        g2a, g2b, g2c, g2d = st.columns(4)
        render_kv(g2a, "Provider Name", robust_get(first_row, ["provider_name","rendering provider","billing provider","provider"]))
        render_kv(g2b, "Provider ID", robust_get(first_row, ["provider_tin","provider id","provider number","npi","provider_no"]))
        render_kv(g2c, "Dependent Number", robust_get(first_row, ["dependent number","dependent id","dep no","dependant id"]))
        render_kv(g2d, "Network", robust_get(first_row, ["network","network type","in network","out of network","ppo"]))
        st.markdown("</div>", unsafe_allow_html=True)

    # Settlement card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Settlement (AI Suggested • Editable)</h4>", unsafe_allow_html=True)

    b1, b2 = st.columns([1,1])
    suggest_clicked = b1.button("Submit", key="submit_suggest")
    journey_clicked = b2.button("Patient Journey", key="journey_btn_inline")
    if journey_clicked:
        st.session_state["show_journey_modal"] = True

    if suggest_clicked:
        ui_rows = []
        # adjudicate every matching line using aca_mod.adjudicate_row (no change to your aca_mod)
        for _, r in rows.iterrows():
            adjudicated = adjudicate_row(r)  # returns dict w/ amounts + rule + rag fields
            ui_rows.append(to_ui_row(r, adjudicated, charge_col_in))
        st.session_state["adjudicated_rows"] = ui_rows

    # Show editable grid (all lines)
    if st.session_state["adjudicated_rows"]:
        df_to_edit = pd.DataFrame(st.session_state["adjudicated_rows"])
        # editor shows only amount fields
        editor_df = df_to_edit[[c for c in ALL_FIELDS if c in df_to_edit.columns]]
        st.data_editor(editor_df, use_container_width=True, hide_index=True, key="editor")

        # Rules + JSON per line
        for i, rec in enumerate(st.session_state["adjudicated_rows"], start=1):
            with st.expander(f"Line {i} • Rule & Reason"):
                st.markdown(f"**Rule:** {rec.get('_rule','—')}")
                st.markdown(f"**Reason:** {rec.get('_reason','—')}")
                if rec.get("_prompt"):
                    with st.expander("RAG Prompt", expanded=False):
                        st.code(rec["_prompt"])
                # JSON from retrieve_hierarchically (parsed via safe_parse_json)
                if rec.get("_json"):
                    with st.expander("RAG JSON", expanded=True):
                        st.json(rec["_json"])
                elif rec.get("_raw"):
                    with st.expander("RAG Raw (not JSON)", expanded=False):
                        st.code(rec["_raw"])

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Patient Journey modal (placeholder)
# -----------------------------
if st.session_state.get("show_journey_modal", False):
    has_modal = callable(getattr(st, "modal", None))
    if has_modal:
        with st.modal("Patient Journey"):
            st.write("This modal will show tables & charts about the patient journey (placeholder).")
            st.line_chart(pd.DataFrame({"Visits":[1,2,3,4,5], "Cost":[120, 300, 180, 260, 210]}).set_index("Visits"))
            st.bar_chart(pd.DataFrame({"Category":["Plan Paid","Employee","Provider"], "Amount":[60,30,10]}).set_index("Category"))
            if st.button("Close"):
                st.session_state["show_journey_modal"] = False
                _safe_rerun()
    else:
        st.info("Patient Journey (fallback panel – update Streamlit to enable modal).")
        st.line_chart(pd.DataFrame({"Visits":[1,2,3,4,5], "Cost":[120, 300, 180, 260, 210]}).set_index("Visits"))
        st.bar_chart(pd.DataFrame({"Category":["Plan Paid","Employee","Provider"], "Amount":[60,30,10]}).set_index("Category"))
        if st.button("Close Journey"):
            st.session_state["show_journey_modal"] = False
            _safe_rerun()

# -----------------------------
# Footer
# -----------------------------
st.caption("© GBT-Claim • UI Prototype")
