from __future__ import annotations

import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
import faiss

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    CLIPModel,
    CLIPTokenizer,
    CLIPImageProcessor,
)

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    # Base directories
    BASE: Path = Path(".")

    # --- Retrieval assets (adjust to your local paths) ---
    # You mentioned these two paths. We only need the *image* index for this demo
    # because retrieval is based on image/text with CLIP. If you also have a text-only
    # index (vector.index), you can wire it similarly.
    IMAGE_INDEX: Path = BASE / "artifacts/Multimodal_NoOptions/indexes/vector_image.index"
    # Optional/unused in minimal demo (kept here for reference):
    # TEXT_INDEX: Path = BASE / "artifacts/Multimodal_NoOptions/indexes/vector.index"

    # The metadata CSV is expected next to the index with suffix _metadata.csv
    # e.g., vector_image.index -> vector_image_metadata.csv

    # --- Fine-tuned CLIP directory (from your training) ---
    CLIP_DIR: Path = BASE / "artifacts/Multimodal_NoOptions/fine_tuned_clip"
    # If you don't have a fine-tuned dir locally, you can fall back to base model:
    CLIP_FALLBACK: str = "openai/clip-vit-base-patch32"

    # --- Generator model (LLM) ---
    # Default: Gemma-3-4B-IT (requires significant VRAM; use device_map="auto")
    LLM_ID: str = "google/gemma-3-4b-it"

    # Retrieval settings
    TOP_K: int = 3

    # Generation settings
    MAX_NEW_TOKENS: int = 128
    TEMPERATURE: float = 0.2


CFG = Config()


# =========================
# UTILITIES
# =========================

def _maybe_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device) if isinstance(x, torch.Tensor) else x


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_faiss_and_metadata(index_path: Path) -> Tuple[faiss.Index, pd.DataFrame, Path]:
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    meta_path = Path(str(index_path).replace(".index", "_metadata.csv"))
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found next to index. Expected: {meta_path}"
        )

    index = faiss.read_index(str(index_path))
    metadata = pd.read_csv(meta_path)

    # Basic sanity checks
    for col in ["image_path", "name", "description"]:
        if col not in metadata.columns:
            raise ValueError(
                f"Metadata must contain column '{col}'. Columns: {list(metadata.columns)}"
            )

    return index, metadata, meta_path


@dataclass
class ClipBundle:
    model: CLIPModel
    tokenizer: CLIPTokenizer
    image_processor: CLIPImageProcessor
    device: torch.device


def load_clip(clip_dir: Path, fallback_id: str) -> ClipBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try local fine-tuned first, otherwise fallback
    if clip_dir.exists():
        model = CLIPModel.from_pretrained(str(clip_dir))
        tokenizer = CLIPTokenizer.from_pretrained(str(clip_dir))
        image_processor = CLIPImageProcessor.from_pretrained(str(clip_dir))
    else:
        model = CLIPModel.from_pretrained(fallback_id)
        tokenizer = CLIPTokenizer.from_pretrained(fallback_id)
        image_processor = CLIPImageProcessor.from_pretrained(fallback_id)

    model.eval().to(device)
    return ClipBundle(model=model, tokenizer=tokenizer, image_processor=image_processor, device=device)


@dataclass
class LlmBundle:
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    model: Gemma3ForConditionalGeneration


def load_llm(llm_id: str) -> LlmBundle:
    # Gemma-3 uses AutoProcessor; keep both tokenizer and processor for flexibility
    processor = AutoProcessor.from_pretrained(llm_id)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        llm_id, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    return LlmBundle(tokenizer=tokenizer, processor=processor, model=model)


# =========================
# RETRIEVAL (CLIP + FAISS)
# =========================

def embed_text(text: str, clip: ClipBundle) -> np.ndarray:
    inputs = clip.tokenizer(text=text, return_tensors="pt")
    inputs = {k: _maybe_to_device(v, clip.device) for k, v in inputs.items()}
    with torch.no_grad():
        feat = clip.model.get_text_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().numpy().astype("float32")


def embed_image(img: Image.Image, clip: ClipBundle) -> np.ndarray:
    inputs = clip.image_processor(images=img, return_tensors="pt")
    inputs = {k: _maybe_to_device(v, clip.device) for k, v in inputs.items()}
    with torch.no_grad():
        feat = clip.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().numpy().astype("float32")


def retrieve(
    index: faiss.Index,
    metadata: pd.DataFrame,
    clip: ClipBundle,
    *,
    text: Optional[str] = None,
    image: Optional[Image.Image] = None,
    top_k: int = 3,
) -> List[Dict]:
    if (text is None or text.strip() == "") and image is None:
        return []

    if image is not None:
        q = embed_image(image, clip)
    else:
        q = embed_text(text.strip(), clip)

    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)

    out: List[Dict] = []
    for i, idx in enumerate(idxs[0]):
        row = metadata.iloc[int(idx)]
        out.append(
            {
                "rank": i + 1,
                "name": row.get("name", ""),
                "description": row.get("description", ""),
                "image_path": str(row.get("image_path", "")),
                "score": float(scores[0][i]),
            }
        )
    return out


# =========================
# PROMPT + GENERATION
# =========================

def build_prompt(question: str, retrieved: List[Dict]) -> str:
    ctx_lines = [
        "Examples from retrieved images and descriptions:",
    ]
    for r in retrieved:
        ctx_lines.append(
            f"- name: {r['name']} | desc: {r['description']} | img: {Path(r['image_path']).name} | score: {r['score']:.4f}"
        )

    ctx_block = "\n".join(ctx_lines)

    prompt = (
        "You are a helpful assistant answering open-ended questions about an image/text query.\n"
        "Use the retrieved examples to ground your answer.\n"
        "Answer succinctly and only output the final answer without extra explanation.\n\n"
        f"{ctx_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


def generate_answer(prompt: str, llm: LlmBundle, max_new_tokens: int, temperature: float) -> str:
    inputs = llm.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(llm.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = llm.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=llm.tokenizer.eos_token_id,
        )
    txt = llm.tokenizer.decode(out[0], skip_special_tokens=True)
    # Keep only the portion after the last "Answer:" to be safe
    if "Answer:" in txt:
        txt = txt.split("Answer:")[-1].strip()
    return txt


# =========================
# GRADIO APP
# =========================
class RagApp:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        # Load models & retrieval assets once at startup
        self.clip = load_clip(cfg.CLIP_DIR, cfg.CLIP_FALLBACK)
        self.llm = load_llm(cfg.LLM_ID)
        self.index, self.metadata, self.meta_path = load_faiss_and_metadata(cfg.IMAGE_INDEX)

    def infer(self, question: str, image: Optional[Image.Image]):
        try:
            clear_gpu_memory()
            retrieved = retrieve(
                self.index,
                self.metadata,
                self.clip,
                text=question,
                image=image,
                top_k=self.cfg.TOP_K,
            )

            prompt = build_prompt(question or "", retrieved)
            answer = generate_answer(
                prompt,
                self.llm,
                max_new_tokens=self.cfg.MAX_NEW_TOKENS,
                temperature=self.cfg.TEMPERATURE,
            )

            # Build simple gallery-style output: [(image, caption), ...]
            gallery_items = []
            for r in retrieved:
                caption = f"{r['rank']}. {r['name']}  (score={r['score']:.3f})"
                img_path = r["image_path"]
                if os.path.exists(img_path):
                    gallery_items.append((img_path, caption))
                else:
                    # If the stored path is not absolute, try relative to metadata's folder
                    alt = str(Path(self.meta_path).parent / Path(img_path).name)
                    if os.path.exists(alt):
                        gallery_items.append((alt, caption))

            # Also return a small dataframe of retrieved rows for debugging
            table = pd.DataFrame(retrieved)
            return answer, gallery_items, table

        except Exception as e:
            return f"Error: {e}", [], pd.DataFrame()


app = RagApp(CFG)

with gr.Blocks(title="Multimodal RAG (minimal)") as demo:
    gr.Markdown("## 🖼️🔎 Minimal Multimodal RAG\nEnter a question and/or drop an image. The system retrieves similar items and generates a short answer.")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Question (optional if image is provided)", placeholder="Who is in this image? What is happening? ...")
            image = gr.Image(type="pil", label="Image (optional if text is provided)")
            submit = gr.Button("Run", variant="primary")
        with gr.Column():
            answer = gr.Textbox(label="Answer", interactive=False)
            gallery = gr.Gallery(label="Top Retrieved", show_label=True, columns=3, height=220)
            table = gr.Dataframe(headers=["rank","name","description","image_path","score"], label="Retrieved (debug)")

    submit.click(fn=app.infer, inputs=[question, image], outputs=[answer, gallery, table])


if __name__ == "__main__":
    # Launch on http://127.0.0.1:7860
    demo.launch()  # set share=True if you want a public link