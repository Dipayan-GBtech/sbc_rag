# aca_mod.py
# --- Wrap of your existing adjudication logic into callable functions ---
# Do not modify the core logic here unless you want behavioral changes.
# Provides: adjudicate_row(row: pd.Series) and adjudicate_claim_by_no(claim_no: str)

import pandas as pd
import re
import json
from typing import Optional, Dict, Any
from rag_engine_mod import retrieve_hierarchically  # Ensure this module exists and is importable

# === Robust JSON Parser ===
def safe_parse_json(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*({.*?})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1)
        else:
            cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        try:
            cleaned = json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}

# === Fallback Regex Parser ===
def parse_rag_response(response_text: str, total_amount: float) -> dict:
    if not response_text or not isinstance(response_text, str):
        return {"plan_paid": round(total_amount * 0.7, 2), "employee_responsibility": round(total_amount * 0.3, 2)}
    cleaned = re.sub(r'\$|\%', '', response_text.lower())
    extracted = {
        "plan_paid": 0.0,
        "co_pay": 0.0,
        "deductible": 0.0,
        "co_insurance": 0.0,
        "employee_responsibility": 0.0,
        "provider_responsibility": 0.0,
    }
    plan_paid = re.search(r'plan\s+pays?[:\s]+([\d.]+)', cleaned)
    copay = re.search(r'copay(?:\s+amount)?[:\s]+([\d.]+)', cleaned)
    deductible = re.search(r'deductible[:\s]+([\d.]+)', cleaned)
    coinsurance = re.search(r'co[\-\s_]*insurance[:\s]+([\d.]+)', cleaned)
    patient_owes = re.search(r'patient\s+owes?[:\s]+([\d.]+)', cleaned)

    if plan_paid:
        extracted["plan_paid"] = float(plan_paid.group(1))
    if copay:
        extracted["co_pay"] = float(copay.group(1))
    if deductible:
        extracted["deductible"] = float(deductible.group(1))
    if coinsurance:
        extracted["co_insurance"] = float(coinsurance.group(1))
    if patient_owes:
        owed = float(patient_owes.group(1))
        known = extracted["co_pay"] + extracted["deductible"] + extracted["co_insurance"]
        extracted["employee_responsibility"] = max(owed - known, 0)

    if sum(extracted.values()) == 0:
        extracted["plan_paid"] = round(total_amount * 0.7, 2)
        extracted["employee_responsibility"] = round(total_amount * 0.3, 2)

    return extracted

# === Load taxid.xlsx if present (special diag codes) ===
special_diag_codes = set()
try:
    taxid_df = pd.read_excel("taxid.xlsx")
    if not taxid_df.empty:
        taxid_df.columns = taxid_df.columns.str.lower().str.strip()
        # attempt to find a sensible column for diag codes
        diag_col = None
        for c in taxid_df.columns:
            if "diag" in c or "code" in c:
                diag_col = c
                break
        if diag_col:
            special_diag_codes = set(taxid_df[diag_col].astype(str).str.lower().str.strip().unique())
except Exception:
    special_diag_codes = set()

# === ACA Rules & Exclusions (load exclusions file) ===
aca_icd_codes = {
    "Z00.00", "Z00.01", "Z00.110", "Z00.111", "Z00.121", "Z00.129", "Z00.3",
    "Z01.411", "Z01.419", "Z11.3", "Z11.4", "Z11.8", "Z11.9", "Z11.51", "Z11.59",
    "Z12.4", "Z12.31", "Z12.32", "Z12.33", "Z12.34", "Z12.35", "Z12.36", "Z12.37",
    "Z12.38", "Z12.39", "Z13.820", "Z13.220", "Z13.1", "Z13.31", "Z13.32", "Z13.39"
}
aca_desc_keywords = {
    "screening", "counseling", "preventive", "wellness", "assessment",
    "well child", "well adult", "immunization", "exam", "checkup"
}

exclusion_keywords = set()
try:
    with open("MEDICAL GENERAL EXCLUSIONS AND LIMITATIONS.txt", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and line.strip()[0].isdigit():
                keyword = line.split(". ", 1)[-1].split(": ")[0].lower().strip()
                exclusion_keywords.add(keyword)
except Exception:
    exclusion_keywords = set()

# === Helper functions used by detection ===
def normalize(text: Any) -> str:
    return str(text).lower().strip() if pd.notna(text) else ""

def is_preventive_desc(desc: str) -> bool:
    desc = normalize(desc)
    return any(re.search(rf'\b{re.escape(kw)}\b', desc) for kw in aca_desc_keywords)

def detect_settlement_rule(row: pd.Series) -> str:
    # Use first 3 diag codes and desc — consistent with your earlier logic
    diag_codes = [normalize(row.get(f"diag{i}", "")) for i in range(1, 4)]
    diag_descrs = [normalize(row.get(f"diag{i}_descr", "")) for i in range(1, 4)]
    # Exclusion check
    for desc in diag_descrs:
        if desc and any(keyword in desc for keyword in exclusion_keywords):
            return "EXCLUSION"
    # ACA check (both code and description)
    for code, desc in zip(diag_codes, diag_descrs):
        if code and code.upper() in aca_icd_codes and is_preventive_desc(desc):
            return "ACA"
    # Special diag codes by taxid list
    for code in diag_codes:
        if code and code.lower() in special_diag_codes:
            return "SPECIAL"
    return "RAG"

def generate_rag_prompt(row: pd.Series) -> str:
    setting = str(row.get("place_of_service", "")).strip().lower()
    if not setting or setting == "facility":
        setting = "inpatient hospitalization"
    proc_code = str(row.get("procedure_code", "")).strip() or "procedure not specified"
    diag_list = []
    # gather diag descrs (up to 12 per your earlier code)
    for i in range(1, 13):
        key = f"diag{i}_descr"
        if key in row.index:
            val = row.get(key)
            if pd.notna(val) and str(val).strip():
                diag_list.append(str(val).strip())
    diag_str = ", ".join(diag_list) if diag_list else "diagnosis not specified"
    network = str(row.get("network_type", "")).strip().lower() or "in-network"
    # deductible status logic copied
    if "deductible_met" in row and pd.notna(row["deductible_met"]) and bool(row["deductible_met"]):
        deductible_status = "deductible and coinsurance have already been met"
    elif "remaining_deductible" in row and pd.notna(row.get("remaining_deductible", None)):
        try:
            deductible_status = f"remaining deductible is ${float(row.get('remaining_deductible', 0)):.2f}"
        except Exception:
            deductible_status = "remaining deductible not specified"
    else:
        deductible_status = "deductible status not specified"
    contracted_amount = row.get("provider_contracted_amount") or row.get("ucr_amount")
    contracted_str = f"UCR amount for this procedure is ${float(contracted_amount):.2f}" if pd.notna(contracted_amount) and contracted_amount != "" else "contracted amount not specified"
    total_charge = float(row.get("charge_amount", 0) or 0)
    prompt = (f"A patient admitted under {setting} with associated procedures {proc_code} "
              f"medically necessary and covered under the plan for a patient diagnosed with {diag_str}, "
              f"when performed at an {network} hospital, the patient’s {deductible_status} "
              f"and {contracted_str}, Total charge is ${total_charge:.2f}, What will be the plan, patient and provider responsibility?")
    return prompt

# === Core adjudication function for a single row ===
def adjudicate_row(row: pd.Series) -> Dict[str, Any]:
    """
    Accepts a pandas Series (single claim row).
    Returns a dict with:
      plan_paid, provider_responsibility, co_pay, deductible, co_insurance, employee_responsibility,
      settlement_rule, settlement_reason_log, rag_prompt, rag_response
    """
    settlement_fields = [
        "plan_paid", "provider_responsibility", "co_pay",
        "deductible", "co_insurance", "employee_responsibility"
    ]
    extracted = {f: 0.0 for f in settlement_fields}
    try:
        rule = detect_settlement_rule(row)
        amount = float(row.get("charge_amount", 0.0) or 0.0)
        log = ""
        prompt = ""
        response = ""

        if rule == "EXCLUSION":
            extracted["employee_responsibility"] = amount
            log = "Denied due to plan exclusion."

        elif rule == "ACA":
            extracted["plan_paid"] = amount
            log = "Covered under ACA preventive services."

        elif rule == "SPECIAL":
            extracted["plan_paid"] = amount
            log = "Covered under TIN-DIAG special rule."

        elif rule == "RAG":
            prompt = generate_rag_prompt(row)
            raw_answer, json_answer, source = retrieve_hierarchically(prompt)
            response = raw_answer

            if json_answer and isinstance(json_answer, dict):
                # mapping tolerant
                extracted["plan_paid"] = float(json_answer.get("PLAN_PAID", json_answer.get("PLAN_PAID_paid", 0)))
                extracted["employee_responsibility"] = float(json_answer.get("EMPLOYEE_RESPONSIBILITY", json_answer.get("EMPLOYEE_RESPONSIBILITY_paid", 0)))
                extracted["provider_responsibility"] = float(json_answer.get("PROVIDER_RESPONSIBILITY", json_answer.get("PROVIDER_RESPONSIBILITY_paid", 0)))
                extracted["co_pay"] = float(json_answer.get("CO_PAY", json_answer.get("CO_PAY_paid", 0)))
                extracted["deductible"] = float(json_answer.get("DEDUCTIBLE", json_answer.get("DEDUCTIABLE_paid", json_answer.get("DEDUCTIBLE_paid", 0))))
                extracted["co_insurance"] = float(json_answer.get("CO_INSURANCE", json_answer.get("CO_INSURANCE_paid", 0)))
                log = f"Settled via RAG (Structured) | {source}"
            else:
                # try safe json parse from raw text
                parsed = safe_parse_json(raw_answer)
                if parsed:
                    extracted["plan_paid"] = float(parsed.get("PLAN_PAID", parsed.get("PLAN_PAID_paid", 0)))
                    extracted["employee_responsibility"] = float(parsed.get("EMPLOYEE_RESPONSIBILITY", parsed.get("EMPLOYEE_RESPONSIBILITY_paid", 0)))
                    extracted["provider_responsibility"] = float(parsed.get("PROVIDER_RESPONSIBILITY", parsed.get("PROVIDER_RESPONSIBILITY_paid", 0)))
                    extracted["co_pay"] = float(parsed.get("CO_PAY", parsed.get("CO_PAY_paid", 0)))
                    extracted["deductible"] = float(parsed.get("DEDUCTIBLE", parsed.get("DEDUCTIABLE_paid", parsed.get("DEDUCTIBLE_paid", 0))))
                    extracted["co_insurance"] = float(parsed.get("CO_INSURANCE", parsed.get("CO_INSURANCE_paid", 0)))
                    log = f"Settled via RAG (JSON Fallback) | {source}"
                else:
                    extracted.update(parse_rag_response(raw_answer, amount))
                    log = f"Settled via RAG (Regex Fallback) | {source}"

            # Enforce sum == charge amount
            total = sum(extracted.values())
            if abs(total - amount) > 0.01:
                if total == 0:
                    extracted["plan_paid"] = round(amount, 2)
                else:
                    factor = amount / total
                    for k in extracted:
                        extracted[k] = round(extracted[k] * factor, 2)
                final_sum = sum(extracted.values())
                if abs(final_sum - amount) > 0.01:
                    diff = round(amount - final_sum, 2)
                    extracted["plan_paid"] = round(extracted["plan_paid"] + diff, 2)
        else:
            # fallback - unexpected rule
            extracted["plan_paid"] = amount
            log = "Fallback: full plan payment."

        return {
            "plan_paid": max(0.0, round(extracted.get("plan_paid", 0.0), 2)),
            "provider_responsibility": max(0.0, round(extracted.get("provider_responsibility", 0.0), 2)),
            "co_pay": max(0.0, round(extracted.get("co_pay", 0.0), 2)),
            "deductible": max(0.0, round(extracted.get("deductible", 0.0), 2)),
            "co_insurance": max(0.0, round(extracted.get("co_insurance", 0.0), 2)),
            "employee_responsibility": max(0.0, round(extracted.get("employee_responsibility", 0.0), 2)),
            "settlement_rule": rule,
            "settlement_reason_log": log,
            "rag_prompt": prompt,
            "rag_response": response
        }

    except Exception as e:
        return {
            "plan_paid": 0.0, "provider_responsibility": 0.0, "co_pay": 0.0, "deductible": 0.0,
            "co_insurance": 0.0, "employee_responsibility": 0.0,
            "settlement_rule": "ERROR",
            "settlement_reason_log": f"Error adjudicating: {e}",
            "rag_prompt": "",
            "rag_response": ""
        }

# === Convenience: adjudicate by claim_no (loads file and finds the row) ===
def adjudicate_claim_by_no(claim_no: str) -> Optional[Dict[str, Any]]:
    df = pd.read_csv("train1_to.csv")
    # normalize column names like the original script
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_", regex=False).str.replace(".", "", regex=False)
    # find claim column
    claim_col = None
    for col in df.columns:
        if "claim" in col and any(x in col for x in ["no", "num", "number", "id"]):
            claim_col = col
            break
    # fallback
    if claim_col is None:
        if "claim_no" in df.columns:
            claim_col = "claim_no"
        else:
            claim_col = df.columns[0]

    matches = df[df[claim_col].astype(str).str.strip().str.lower() == str(claim_no).strip().lower()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    result = adjudicate_row(row)
    # add original charge amount for UI convenience
    result["charge_amount"] = float(row.get("charge_amount", 0) or 0)
    result["claim_no"] = row.get(claim_col)
    return result
