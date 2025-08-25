# rag_engine_mod.py
import os
import json
import re
import numpy as np
import requests
import fitz  # PyMuPDF (used if you index PDFs)
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- Config (from environment with fallbacks) ----------------
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M")
#MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:30b-a3b-q4_K_M")
#MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-r1:7b-qwen-distill-q4_K_M")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY","")
#MODEL_API_URL=os.environ.get("MODEL_API_URL", "https://api.together.xyz/v1/chat/completions")
#MODEL_NAME=os.environ.get("MODEL_NAME", "Qwen/Qwen2-72B-Instruct")
#MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "019f4b463649d0f2cb6e13198e8b7547974c3de9e7442f56145400d843661be7")  # optional
#MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "0dc9915774eb44e28d782363e73456063c2cc2fab6dc492c9c28063384217c8d")

# Data/index locations (adjust if needed)
DATA_DIR = "data"
BASE_KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")
PLAN_DOC_DIR = os.path.join(DATA_DIR, "plan_docs")
SBC_RULE_PATH = os.path.join(DATA_DIR, "BBI_Plan_Final.json")
INDEX_BASE_PATH = os.path.join(DATA_DIR, "faiss_index_base.bin")
INDEX_PLAN_PATH = os.path.join(DATA_DIR, "faiss_index_plan.bin")

# Embedding model for indexing/search
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "multi-qa-MiniLM-L6-cos-v1")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ---------------- Helpers ----------------
def find_first_json_block(text: str):
    """Return the first balanced JSON object substring or None."""
    if not text or not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    brace = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                return text[start:i+1]
    return None

def safe_parse_json(text: str):
    """
    Extract and parse the first JSON object from text.
    Returns dict on success or {} on failure.
    """
    if not text or not isinstance(text, str):
        return {}
    # Remove leading/trailing whitespace and common code fences
    cleaned = text.strip()
    # Remove triple backticks if present
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("` \n")
    # Unwrap quoted JSON like: "{"a":1}"
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try finding first balanced JSON block
    candidate = find_first_json_block(cleaned)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Last-resort regex (non-greedy)
    match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}

# ---------------- PDF indexing helpers (optional) ----------------
def extract_text_from_pdfs(pdf_folder):
    all_text = []
    metadata = []
    if not os.path.exists(pdf_folder):
        return [], []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            try:
                doc = fitz.open(path)
            except Exception as e:
                print(f"Failed to open PDF {path}: {e}")
                continue
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text and text.strip():
                    all_text.append(text)
                    metadata.append({"filename": filename, "page": page_num + 1})
    return all_text, metadata

def chunk_text(text_list, max_chars=1000, overlap=200):
    chunks = []
    for text in text_list:
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start += max_chars - overlap
    return chunks

def build_index_for_directory(pdf_dir, index_path):
    raw_text, meta = extract_text_from_pdfs(pdf_dir)
    chunks = chunk_text(raw_text)
    print(f"Indexed {len(chunks)} chunks from {pdf_dir}")
    if not chunks:
        return None, [], []
    if os.path.exists(index_path):
        try:
            return faiss.read_index(index_path), chunks, meta
        except Exception:
            pass
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    try:
        faiss.write_index(idx, index_path)
    except Exception:
        pass
    return idx, chunks, meta

# Build indexes (safe)
index_base, chunks_base, meta_base = build_index_for_directory(BASE_KNOWLEDGE_DIR, INDEX_BASE_PATH)
index_plan, chunks_plan, meta_plan = build_index_for_directory(PLAN_DOC_DIR, INDEX_PLAN_PATH)

# ---------------- SBC JSON search (optional) ----------------
def search_sbc_json(query):
    """
    If you have an SBC JSON file, try to match the query to a rule.
    Returns either (raw_answer_string, parsed_json_dict) or None.
    """
    if not os.path.exists(SBC_RULE_PATH):
        return None
    try:
        with open(SBC_RULE_PATH, "r", encoding="utf-8") as f:
            rules = json.load(f)
        for rule in rules:
            desc = rule.get("description", "") if isinstance(rule, dict) else ""
            if desc and query.lower() in desc.lower():
                ans = rule.get("answer")
                if isinstance(ans, dict):
                    return json.dumps(ans), ans
                if isinstance(ans, str):
                    return ans, safe_parse_json(ans)
    except Exception as e:
        print("SBC search error:", e)
    return None

# ---------------- LLM call (handles streaming-style fragments) ----------------
def generate_with_together(prompt: str, api_url: str = None, model_name: str = None, timeout: int = 720):
    """
    Calls the LLM endpoint and returns the assembled assistant content string.
    Supports:
      - standard JSON response (choices[0].message.content)
      - newline-delimited JSON event lines (each with 'message' containing partial 'content')
    """
    if api_url is None:
        api_url = MODEL_API_URL
    if model_name is None:
        model_name = MODEL_NAME

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 800
    }

    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        return f"⚠️ Error: request failed: {e}"

    if resp.status_code != 200:
        # Return textual error for debugging
        return f"⚠️ API Error: {resp.status_code} - {resp.text}"

    # Try structured JSON first
    try:
        data = resp.json()
    except ValueError:
        # Non-JSON body or newline JSON events (assemble fragments)
        assembled = ""
        text = resp.text or ""
        for line in text.splitlines():
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                obj = json.loads(line_strip)
            except Exception:
                # skip non-JSON lines
                continue
            # extract content from common locations
            msg = None
            if isinstance(obj, dict):
                if "message" in obj and isinstance(obj["message"], dict):
                    msg = obj["message"].get("content")
                if msg is None and "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                    c0 = obj["choices"][0]
                    if isinstance(c0, dict):
                        if "message" in c0 and isinstance(c0["message"], dict):
                            msg = c0["message"].get("content")
                        elif "delta" in c0 and isinstance(c0["delta"], dict):
                            msg = c0["delta"].get("content")
                        elif "text" in c0:
                            msg = c0.get("text")
            if msg:
                assembled += str(msg)
        assembled = assembled.strip()
        # final fallback: if nothing assembled, return raw text for debugging
        return assembled if assembled else resp.text

    # data is a dict - extract common response forms
    if isinstance(data, dict):
        # Chat-style: choices[0].message.content
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            c0 = data["choices"][0]
            if isinstance(c0, dict):
                # message.content
                if "message" in c0 and isinstance(c0["message"], dict) and "content" in c0["message"]:
                    return c0["message"]["content"].strip()
                # choices[0].text
                if "text" in c0 and isinstance(c0["text"], str):
                    return c0["text"].strip()
                # delta stream
                if "delta" in c0 and isinstance(c0["delta"], dict) and "content" in c0["delta"]:
                    return c0["delta"]["content"].strip()
        # Some servers return 'response' or 'result'
        if "response" in data:
            return str(data["response"]).strip()
        if "result" in data:
            return str(data["result"]).strip()
        # else, return JSON string
        try:
            return json.dumps(data)
        except Exception:
            return str(data)
    return str(data)

# ---------------- RAG retrieval functions ----------------
def retrieve_from_index(query, index, chunks, metadata, k=5):
    import re
    import json
    if not index or not chunks:
        return "I don't know.", {}, "", None

    # Extract charge amount
    charge_amount = None
    m = re.search(r"Total charge is \$?([\d,]+(?:\.\d{1,2})?)", query)
    if m:
        try:
            charge_amount = float(m.group(1).replace(",", ""))
        except:
            pass

    # Embed and search
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb), k=k)
    contexts, sources = [], []
    for i in range(min(3, k)):
        idx = int(I[0][i])
        if idx == -1 or idx >= len(chunks) or idx >= len(metadata):
            continue
        contexts.append(chunks[idx])
        sources.append(metadata[idx])
    if not contexts:
        return "I don't know.", {}, "", None
    full_context = "\n\n---\n\n".join(contexts)

    # Enhanced prompt with strict formatting and math enforcement
    prompt = f"""
You are an insurance claim adjudication system. Based on the rules below, calculate financial responsibilities for this claim.

Rules:
{full_context}

Claim:
{query}

INSTRUCTIONS:
- Return ONLY a valid JSON object.
- Use EXACTLY these 6 keys:
  "PLAN_PAID_paid", "PROVIDER_RESPONSIBILITY_paid", "CO_PAY_paid", "DEDUCTIBLE_paid", "CO_INSURANCE_paid", "EMPLOYEE_RESPONSIBILITY_paid"
- All values must be **numbers only** (floats). No formulas, no comments, no text.
- The sum of all 6 values must exactly equal the Total Charge: ${charge_amount:.2f}.
- If a field does not apply, use 0.0.
- Never invent new keys or use variations.

Return only the JSON object.
"""

    raw = generate_with_together(prompt)
    print("\n--- RAW RAG ANSWER ---")
    print(repr(raw))
    print("-----------------------\n")

    # Normalize raw string
    cleaned = re.sub(r"//.*|/\*.*?\*/|<!--.*?-->", "", raw, flags=re.DOTALL)  # Remove comments
    cleaned = re.sub(r"```json|```", "", cleaned)  # Remove code blocks
    cleaned = cleaned.strip()

    parsed = {}
    final_keys = [
        "PLAN_PAID_paid",
        "PROVIDER_RESPONSIBILITY_paid",
        "CO_PAY_paid",
        "DEDUCTIBLE_paid",  # Fixed spelling
        "CO_INSURANCE_paid",
        "EMPLOYEE_RESPONSIBILITY_paid"
    ]

    # Try JSON extraction
    try:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group(0))
            # Normalize keys
            key_map = {k.lower().replace("_", "").replace(" ", ""): k for k in final_keys}
            for k, v in obj.items():
                norm_k = k.lower().replace("_", "").replace(" ", "")
                if norm_k in key_map:
                    try:
                        parsed[key_map[norm_k]] = float(v)
                    except (TypeError, ValueError):
                        parsed[key_map[norm_k]] = 0.0
    except Exception as e:
        print(f"JSON parse failed: {e}")
        # Fallback: extract all numbers and assign in order
        nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
        nums = [float(x) for x in nums if float(x) >= 0]
        if len(nums) >= len(final_keys):
            nums = nums[:len(final_keys)]
        else:
            nums = nums + [0.0] * (len(final_keys) - len(nums))
        parsed = {k: nums[i] for i, k in enumerate(final_keys)}

    # Ensure all keys exist
    for k in final_keys:
        parsed.setdefault(k, 0.0)

    # Enforce sum = charge_amount
    if charge_amount is not None:
        total = sum(parsed.values())
        if abs(total - charge_amount) > 1e-2:  # Not close
            if total == 0:
                # Default to plan pays all
                parsed["PLAN_PAID_paid"] = round(charge_amount, 2)
            else:
                # Scale all values proportionally
                factor = charge_amount / total
                for k in parsed:
                    parsed[k] = round(parsed[k] * factor, 2)
            # Final adjustment to prevent float rounding errors
            final_sum = sum(parsed.values())
            if abs(final_sum - charge_amount) > 1e-2:
                # Adjust one field (e.g., PLAN_PAID) to fix rounding
                diff = round(charge_amount - final_sum, 2)
                parsed["PLAN_PAID_paid"] = round(parsed["PLAN_PAID_paid"] + diff, 2)

    return raw, parsed, full_context, (sources[0] if sources else None)

def retrieve_hierarchically(query):
    """
    Returns (raw_text, parsed_json_dict, source_label)
    """
    # SBC JSON rules
    sbc = search_sbc_json(query)
    if sbc:
        raw, parsed = sbc if isinstance(sbc, tuple) else (sbc, safe_parse_json(sbc))
        return raw, parsed, "SBC JSON"

    # Base knowledge
    raw_base, parsed_base, ctx_base, src_base = retrieve_from_index(query, index_base, chunks_base, meta_base)
    if parsed_base and any(v != 0 for v in parsed_base.values()):
        return raw_base, parsed_base, "Base Knowledge PDF"

    # Plan document
    raw_plan, parsed_plan, ctx_plan, src_plan = retrieve_from_index(query, index_plan, chunks_plan, meta_plan)
    return raw_plan, parsed_plan, "Plan Document PDF"


