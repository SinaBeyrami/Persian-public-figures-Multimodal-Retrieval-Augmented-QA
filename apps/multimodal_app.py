from dotenv import load_dotenv
load_dotenv()

import os, io, json, base64, argparse, textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import gradio as gr
import torch

# ================== Paths ==================
BASE_DIR       = Path("artifacts/multimodal")
DOCS_PATH      = BASE_DIR / "embeddings/docs.jsonl"
CLIP_TEXT_ROWS = BASE_DIR / "embeddings/clip_text_rows.json"
FAISS_TEXT_IDX = BASE_DIR / "indexes/faiss_text.idx"
FAISS_IMG_IDX  = BASE_DIR / "indexes/faiss_image.idx"
CLIP_TEXT_IDX  = BASE_DIR / "indexes/clip_text.idx"  # optional

# ================== Environment & API Keys ==================
MODEL_NAME     = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
GAPGPT_API_KEY = os.getenv("GAPGPT_API_KEY")

# ================== Globals ==================
text_model = None
clip_model = None
clip_preprocess = None
clip_tokenizer = None
text_index = None
img_index = None
clip_text_index = None
docs : List[Dict[str,Any]] = []
row_map : Dict[int,int] = {}
USE_FAISS = False

# ================== Utils ==================

def read_jsonl(path: Path) -> List[Dict[str,Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def try_import_faiss():
    global USE_FAISS
    try:
        import faiss  # type: ignore
        USE_FAISS = True
    except Exception:
        USE_FAISS = False

def load_faiss(path: Path):
    import faiss  # type: ignore
    return faiss.read_index(str(path)) if path.exists() else None

def knn(index, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    if index is None: return np.array([]), np.array([])
    D, I = index.search(q.reshape(1,-1).astype("float32"), k)
    return D[0], I[0]

def b64_image(img: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ================== Load models & indexes ==================

def load_models():
    global text_model, clip_model, clip_preprocess, clip_tokenizer
    global text_index, img_index, clip_text_index, docs, row_map

    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    import open_clip

    try_import_faiss()
    e5_dir = snapshot_download("intfloat/multilingual-e5-base", local_dir="artifacts/multimodal/m-e5-base", local_dir_use_symlinks=False, resume_download=True)
    _ = snapshot_download("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", local_dir="artifacts/multimodal/openclip-b32", local_dir_use_symlinks=False, resume_download=True)

    text_model = SentenceTransformer(e5_dir, device="cpu")
    clip_model_, _, clip_preprocess_ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model = clip_model_.to("cpu").eval()
    clip_preprocess = clip_preprocess_
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    docs[:] = read_jsonl(DOCS_PATH)
    for d in docs:
        if "text_row" in d:
            row_map[int(d["text_row"])] = len(row_map)

    if USE_FAISS:
        import faiss
        text_index = load_faiss(FAISS_TEXT_IDX)
        img_index  = load_faiss(FAISS_IMG_IDX)
        if CLIP_TEXT_IDX.exists():
            clip_text_index = load_faiss(CLIP_TEXT_IDX)

# ================== Retrieval ==================

def search(query: str, k: int = 5, alpha: float = 0.6) -> List[Dict[str,Any]]:
    if text_index is None and img_index is None:
        return []
    t_hits, i_hits = {}, {}
    if text_index is not None:
        q = text_model.encode(f"query: {query}", normalize_embeddings=True)
        D,I = knn(text_index, q, k*5)
        scores = (D - D.min())/(D.ptp()+1e-6)
        t_hits = {int(i): float(s) for i,s in zip(I,scores)}
    if img_index is not None:
        with torch.no_grad():
            toks = clip_tokenizer([query]).to("cpu")
            v = clip_model.encode_text(toks)
            v = v / v.norm(dim=-1, keepdim=True)
        D,I = knn(img_index, v.cpu().numpy()[0], k*5)
        scores = (D - D.min())/(D.ptp()+1e-6)
        i_hits = {row_map.get(int(i), -1): float(s) for i,s in zip(I,scores) if int(i) in row_map}
    combined = {}
    for i,s in t_hits.items(): combined[i] = combined.get(i,0)+alpha*s
    for i,s in i_hits.items(): combined[i] = combined.get(i,0)+(1-alpha)*s
    ranked = sorted([(i,s) for i,s in combined.items() if i>=0], key=lambda x:x[1], reverse=True)[:k]
    out = []
    for i,s in ranked:
        d = dict(docs[i]); d['_score']=s; out.append(d)
    return out

# ================== Answer generation (GapGPT) ==================

def answer(query: str, hits: List[Dict[str,Any]], img: np.ndarray|None) -> str:
    context = "\n\n".join([f"نام: {h['name']}\nخلاصه: {h['summary']}" for h in hits])
    if not GAPGPT_API_KEY:
        return context if context else "هیچ نتیجه‌ای پیدا نشد."
    from openai import OpenAI
    client = OpenAI(base_url="https://api.gapgpt.app/v1", api_key=GAPGPT_API_KEY)
    content=[{"role":"system","content":"You are a helpful assistant. Answer ONLY in Persian."},
             {"role":"user","content": f"پرسش: {query}\n{context}"}]
    if img is not None:
        content.insert(1,{"role":"user","content": {"type":"image_url","image_url":{"url":b64_image(img),"detail":"low"}}})
    resp = client.chat.completions.create(model=MODEL_NAME, messages=content, temperature=0)
    return resp.choices[0].message.content.strip()

# ================== Gradio UI ==================

def pipeline(q:str, img:np.ndarray|None, k:int, alpha:float):
    if not q.strip():
        return "لطفاً سوال وارد کنید", [], "[]"
    hits = search(q,k,alpha)
    ans = answer(q,hits,img)
    gallery=[(h.get('image_path'), f"{h['name']} ({h.get('_score',0):.2f})") for h in hits]
    return ans, gallery, json.dumps(hits, ensure_ascii=False, indent=2)

def build_ui():
    with gr.Blocks(title="Multimodal RAG Demo") as demo:
        gr.Markdown("## 🎯 دمو RAG چندرسانه‌ای (متن + تصویر)")
        with gr.Row():
            with gr.Column():
                q = gr.Textbox(label="پرسش", lines=2)
                img = gr.Image(type="numpy", label="تصویر اختیاری")
                k = gr.Slider(1,10,5,step=1,label="k")
                alpha = gr.Slider(0.0,1.0,0.6,step=0.05,label="α وزن متن")
                btn = gr.Button("🔍 جستجو")
            with gr.Column():
                out = gr.Markdown()
                gal = gr.Gallery(label="اسناد بازیابی‌شده").style(grid=2)
                meta= gr.Code(label="جزئیات")
        btn.click(pipeline,[q,img,k,alpha],[out,gal,meta])
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    load_models()
    build_ui().launch(server_name="0.0.0.0", server_port=args.port)
