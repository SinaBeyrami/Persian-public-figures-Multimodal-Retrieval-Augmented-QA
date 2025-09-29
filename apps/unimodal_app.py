from dotenv import load_dotenv
load_dotenv()

import os, json, re, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import gradio as gr

# ================== Paths ==================
BASE           = Path(".")
DOCS_MAP       = Path("data/processed/docs.jsonl")
TEXT_INDEX     = Path("artifacts/unimodal/indexes/faiss_text.idx")
HNSW_BIN       = Path("artifacts/unimodal/indexes/faiss_text.bin")   # optional
RUNS_DIR       = Path("artifacts/unimodal/runs")                      # optional - just for logging
E5_LOCAL_DIR   = Path("models/m-e5-base")                             # if exists, load from local, else download it
E5_REPO        = "intfloat/multilingual-e5-base"

MODEL_NAME     = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
GAPGPT_API_KEY = os.getenv("GAPGPT_API_KEY")

# ================== Normalization helpers ==================
ARABIC_TO_PERSIAN = str.maketrans({"ي":"ی","ك":"ک","ة":"ه","ۀ":"ه","ؤ":"و","إ":"ا","أ":"ا","ٱ":"ا","ى":"ی"})
EASTERN_DIGITS = str.maketrans({
    "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
    "٠":"0","١":"1","٢":"2","٣":"3","٤":"4","٥":"5","٦":"6","٧":"7","٨":"8","٩":"9",
})
PUNCT_RE = re.compile(r"[\u200c\u200f\u202a-\u202e\s\-\_]+")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.translate(ARABIC_TO_PERSIAN)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(EASTERN_DIGITS)
    s = PUNCT_RE.sub(" ", s).strip().casefold()
    return s

def normalize_name_field(x):
    if isinstance(x, list):
        for v in x:
            if isinstance(v, str) and v.strip():
                return v
        return ""
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)

def normalize_text_field(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = [v.strip() for v in x if isinstance(v, str) and v.strip()]
        return " | ".join(parts)
    if isinstance(x, dict):
        parts = []
        for v in x.values():
            s = normalize_text_field(v)
            if s:
                parts.append(s)
        return " | ".join(parts)
    return "" if x is None else str(x)

# ================== Load docs ==================
if not DOCS_MAP.exists():
    raise FileNotFoundError(f"Missing {DOCS_MAP.resolve()}")
docs = [json.loads(line) for line in DOCS_MAP.open("r", encoding="utf-8")]

# ================== Load encoder (E5) ==================
from sentence_transformers import SentenceTransformer
e5_path = str(E5_LOCAL_DIR) if E5_LOCAL_DIR.exists() else E5_REPO
text_model = SentenceTransformer(e5_path, device="cpu")

# ================== Load index (FAISS / HNSW) ==================
USE_FAISS = False
USE_HNSW  = False
index = None
dim = 768  # multilingual-e5-base

try:
    import faiss
    if TEXT_INDEX.exists():
        index = faiss.read_index(str(TEXT_INDEX))
        USE_FAISS = True
except Exception:
    pass

if index is None and HNSW_BIN.exists():
    try:
        import hnswlib
        hindex = hnswlib.Index(space="cosine", dim=dim)
        hindex.load_index(str(HNSW_BIN))
        hindex.set_ef(64)
        index = hindex
        USE_HNSW = True
    except Exception:
        pass

if index is None:
    raise FileNotFoundError(
        "No vector index found. Provide either:\n"
        f" - {TEXT_INDEX}\n"
        f" - or {HNSW_BIN}"
    )

def knn(index, q, k=5) -> Tuple[np.ndarray, np.ndarray]:
    if USE_FAISS:
        D, I = index.search(q.reshape(1,-1).astype("float32"), k)
        return D[0], I[0]
    else:
        I, D = index.knn_query(q.reshape(1,-1).astype("float32"), k)
        return (1.0 - D[0]), I[0]  # cosine sim

# ================== Retrieval ==================
def search(query: str, k: int = 5):
    if not query.strip():
        return []
    q_vec = text_model.encode(f"query: {query}",
                              convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
    sims, idxs = knn(index, q_vec, k=int(k))
    out = []
    for s, i in zip(sims, idxs):
        i = int(i)
        if 0 <= i < len(docs):
            rec = dict(docs[i])
            rec["_score"] = float(s)
            out.append(rec)
    # optional: search logging
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with (RUNS_DIR / "search_logs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"q": query, "k": int(k),
                                "top": [{"name": r.get("name",""), "score": r.get("_score",0.0)} for r in out[:3]]},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass
    return out

# ================== Generative (OpenAI-compatible) ==================
from openai import OpenAI
client = OpenAI(base_url="https://api.gapgpt.app/v1", api_key=GAPGPT_API_KEY) if GAPGPT_API_KEY else None

def trim(s: str, max_chars: int = 1400) -> str:
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars] + "…")

def build_prompt_persian(question: str, options: List[str], ctxs: List[Dict[str,str]]) -> List[Dict[str,str]]:
    k = max(1, len(ctxs))
    task = (
        "تو یک مولد پاسخ مبتنی بر RAG هستی. فقط از اسناد بازیابی‌شده استفاده کن و اگر پاسخ در آن‌ها نبود، "
        "predicted_option را خالی بگذار.\n"
        "خروجی فقط JSON با کلیدهای زیر باشد:\n"
        "{\n"
        '  "target_name": "نام کامل یا خالی",\n'
        '  "predicted_option": "یکی از گزینه‌ها یا خالی",\n'
        '  "evidence_span": "نقل‌قول مستقیم از یکی از منابع"\n'
        "}\n"
    )
    q_block = "سؤال:\n" + question.strip()
    if options:
        q_block += "\n\nگزینه‌ها:\n" + "\n".join([f"- {o}" for o in options])

    ctx_lines = []
    for i, c in enumerate(ctxs, start=1):
        nm = (c.get("name") or "").strip()
        sm = trim((c.get("summary") or "").strip())
        header = f"[{i}] نام: {nm}" if nm else f"[{i}]"
        body = sm if sm else "—"
        ctx_lines.append(f"{header}\n{body}")
    ctx_block = "\n\n".join(ctx_lines) if ctx_lines else "هیچ متنی موجود نیست."

    user_content = (
        f"{task}\n\n{q_block}\n\n---\n"
        f"اسناد بازیابی‌شده (top-{k}):\n{ctx_block}\n\nفقط JSON پاسخ بده."
    )
    return [
        {"role":"system","content":"You are a helpful assistant. Answer ONLY in JSON."},
        {"role":"user","content": user_content}
    ]

def call_model(messages: List[Dict[str,str]]) -> Dict[str,Any]:
    if client is None:
        return {"target_name":"","predicted_option":"","evidence_span":"(LLM API key not set)"}
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0,
            response_format={"type":"json_object"}
        )
        txt = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"target_name":"","predicted_option":"","evidence_span": f"LLM error: {e}"}

    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL)
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        obj = json.loads(m.group(0)) if m else {"target_name":"","predicted_option":"","evidence_span":txt}
    obj.setdefault("target_name","")
    obj.setdefault("predicted_option","")
    obj.setdefault("evidence_span","")
    return obj

def hits_to_df(hits: List[Dict[str,Any]]) -> pd.DataFrame:
    rows = []
    for j, h in enumerate(hits, start=1):
        rows.append({
            "rank": j,
            "name": h.get("name",""),
            "score": round(float(h.get("_score",0.0)), 3),
            "summary_snippet": (h.get("summary","") or "")[:300]
        })
    return pd.DataFrame(rows, columns=["rank","name","score","summary_snippet"])

def build_ctx_from_hits(hits: List[Dict[str,Any]], top_m: int) -> List[Dict[str,str]]:
    return [{"name": h.get("name",""), "summary": h.get("summary","") or ""} for h in hits[:top_m]]

# ================== Gradio Callbacks ==================
def ui_retrieve(query: str, top_k: int):
    try:
        hits = search(query, k=int(top_k)) or []
        df = hits_to_df(hits)
        return df, gr.update(value=f"{len(df)} results", visible=True)
    except Exception as e:
        return pd.DataFrame(), gr.update(value=f"Error: {e}", visible=True)

def ui_generate(query: str, top_k: int, options_csv: str):
    try:
        opts = []
        if options_csv.strip():
            opts = [o.strip() for o in options_csv.split(",") if o.strip()]
        hits = search(query, k=int(top_k)) or []
        ctxs = build_ctx_from_hits(hits, top_m=int(top_k)) or [{"name":"", "summary":""}]
        messages = build_prompt_persian(query, opts, ctxs)
        out = call_model(messages)
        # align predicted_option to opts
        po = str(out.get("predicted_option","")).strip()
        if po and opts and po not in opts:
            opts_norm = [normalize_text(o) for o in opts]
            po_norm   = normalize_text(po)
            if po_norm in opts_norm:
                out["predicted_option"] = opts[opts_norm.index(po_norm)]
        ctx_pretty = "\n\n".join([f"— {c.get('name','')}\n{(c.get('summary','') or '')[:600]}" for c in ctxs])
        return json.dumps(out, ensure_ascii=False, indent=2), ctx_pretty
    except Exception as e:
        err = {"target_name":"","predicted_option":"","evidence_span": f"Error: {e}"}
        return json.dumps(err, ensure_ascii=False, indent=2), ""

# ================== Gradio UI ==================
with gr.Blocks(title="Unimodal RAG Demo") as demo:
    gr.Markdown("## Unimodal RAG (Text) — Demo")

    with gr.Tab("Retrieve"):
        q1 = gr.Textbox(label="Query (پرسش)", placeholder="مثلاً: قهرمان مشهور کشتی آزاد ایران")
        k1 = gr.Slider(1, 10, value=5, step=1, label="top_k")
        btn1 = gr.Button("Retrieve")
        out_tbl = gr.Dataframe(headers=["rank","name","score","summary_snippet"], interactive=False)
        status1 = gr.Markdown(visible=False)
        btn1.click(ui_retrieve, inputs=[q1, k1], outputs=[out_tbl, status1])

    with gr.Tab("Generate"):
        q2 = gr.Textbox(label="Query (پرسش)")
        k2 = gr.Slider(1, 10, value=5, step=1, label="top_k")
        opts = gr.Textbox(label="Options (اختیاری، با ویرگول جدا کن)",
                          placeholder="مثلاً: تختی, پهلوان, ...")
        btn2 = gr.Button("Generate")
        out_json = gr.Code(label="Model JSON", language="json")
        out_ctxs = gr.Textbox(label="Contexts used", lines=12)
        btn2.click(ui_generate, inputs=[q2, k2, opts], outputs=[out_json, out_ctxs])

    gr.Markdown(
        f"**Docs:** `{DOCS_MAP}`  •  **Index:** "
        f"`{TEXT_INDEX if TEXT_INDEX.exists() else (HNSW_BIN if HNSW_BIN.exists() else 'N/A')}`  •  "
        f"**LLM set:** `{bool(GAPGPT_API_KEY)}`"
    )

if __name__ == "__main__":
    try:
        demo.queue() 
    except TypeError:
        pass
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
