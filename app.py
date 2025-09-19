Multilingual Text → English (Translate + Summarize + Keywords)
Run:
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import gradio as gr
from langdetect import detect, DetectorFactory
from transformers import pipeline
from functools import lru_cache

# Keyword extraction
import yake

# Make langdetect deterministic
DetectorFactory.seed = 42

# Helpers: device & chunking
def _auto_device() -> int | str:
    """
    Return device map for HF pipelines:
      - CUDA id (0) if GPU is available
      - 'cpu' otherwise
    """
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _clean_text(text: str) -> str:
    """Basic cleanup to avoid weird whitespace and HTML remnants."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_into_chunks(text: str, words_per_chunk: int = 180) -> List[str]:
    """
    Split long text into manageable chunks for summarization.
    This is word-based (simple & robust).
    """
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        chunks.append(chunk)
    return chunks


# -----------------------------
# Lazy / cached pipelines
# -----------------------------
@lru_cache(maxsize=1)
def get_summarizer():
    """English summarizer (fast DistilBART)."""
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=_auto_device(),
    )


@lru_cache(maxsize=1)
def get_translator_es_en():
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-es-en",
        device=_auto_device(),
    )


@lru_cache(maxsize=1)
def get_translator_de_en():
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-de-en",
        device=_auto_device(),
    )


@lru_cache(maxsize=1)
def get_translator_ar_en():
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-ar-en",
        device=_auto_device(),
    )


# -----------------------------
# Core processing
# -----------------------------
SUPPORTED = {"en", "es", "de", "ar"}
LANG_NAME = {"en": "English", "es": "Spanish", "de": "German", "ar": "Arabic"}


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED else "en"
    except Exception:
        return "en"


def translate_to_english(text: str, lang: str) -> str:
    if lang == "en":
        return text

    try:
        if lang == "es":
            trans = get_translator_es_en()(text, max_length=512)
        elif lang == "de":
            trans = get_translator_de_en()(text, max_length=512)
        elif lang == "ar":
            trans = get_translator_ar_en()(text, max_length=512)
        else:
            # Fallback: return original if unsupported
            return text

        if isinstance(trans, list) and len(trans) and "translation_text" in trans[0]:
            return trans[0]["translation_text"]
        return text
    except Exception:
        # If translation fails, return original text
        return text


def summarize_english(text: str, max_tokens: int = 130, min_tokens: int = 30) -> str:
    text = _clean_text(text)
    if not text:
        return ""

    chunks = _split_into_chunks(text, words_per_chunk=180)
    summarizer = get_summarizer()

    summaries = []
    for ch in chunks:
        try:
            out = summarizer(
                ch,
                max_length=max_tokens,
                min_length=min_tokens,
                do_sample=False,
            )
            if isinstance(out, list) and len(out) and "summary_text" in out[0]:
                summaries.append(out[0]["summary_text"])
        except Exception:
            # If one chunk fails, skip it
            continue

    return " ".join(summaries).strip() if summaries else ""


def extract_keywords_english(text: str, top_k: int = 10) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    kw = yake.KeywordExtractor(lan="en", n=1, top=top_k)
    try:
        candidates = kw.extract_keywords(text)
        # candidates is list of (term, score) → lower score = more important
        return [term for term, _ in sorted(candidates, key=lambda x: x[1])]
    except Exception:
        return []


def process_text(
    text: str, max_summary_len: int = 130, kw_top_k: int = 8
) -> Tuple[str, str, str, List[str]]:
    """
    Full pipeline:
      - detect language
      - translate to English (if needed)
      - summarize in English
      - extract English keywords
    Returns: (detected_language_verbose, english_text, summary, keywords)
    """
    text = _clean_text(text)
    if not text:
        return "English", "", "", []

    lang = detect_language(text)
    english_text = translate_to_english(text, lang)
    summary = summarize_english(english_text, max_tokens=max_summary_len)
    keywords = extract_keywords_english(english_text, top_k=kw_top_k)

    detected = LANG_NAME.get(lang, "English")
    return detected, english_text, summary, keywords


# -----------------------------
# Gradio UI
# -----------------------------
EXAMPLE_INPUTS = [
    "La inflación anual en España cayó al 2,3% en junio, impulsada por los precios de la energía.",
    "Die Europäische Zentralbank erwägt, die Zinssätze stabil zu halten, da sich die Inflation verlangsamt.",
    "أعلنت الشركة عن أرباح فاقت التوقعات بسبب زيادة الطلب على المنتجات الجديدة.",
]

with gr.Blocks(title="Multilingual Text → English") as demo:
    gr.Markdown(
        """
# 🌍 Multilingual Text → English (Translate + Summarize + Keywords)

Paste **any text** in Spanish / German / Arabic / English.  
This app:
- detects the language,
- translates non-English → English (Helsinki-NLP),
- summarizes in English (DistilBART CNN),
- extracts keywords (YAKE).
"""
    )

    with gr.Row():
        inp = gr.Textbox(
            lines=8,
            label="Paste text (es/de/ar/en)",
            placeholder="Paste an article or paragraph here…",
        )

    with gr.Row():
        max_len = gr.Slider(
            minimum=60,
            maximum=200,
            value=130,
            step=10,
            label="Max summary length (tokens)",
        )
        topk = gr.Slider(
            minimum=5, maximum=20, value=8, step=1, label="Keywords (top-k)"
        )

    with gr.Row():
        detected_lang = gr.Textbox(label="Detected language", interactive=False)
        english_out = gr.Textbox(label="English text", lines=8)
    summary_out = gr.Textbox(label="Summary (English)", lines=6)
    keywords_out = gr.HighlightedText(
        label="Keywords (English)",
        combine_adjacent=True,
    )

    run_btn = gr.Button("Translate & Summarize", variant="primary")

    def _ui_wrapper(text, max_summary_len, top_k):
        det, en, summ, kws = process_text(text, max_summary_len, top_k)
        # Highlight keywords in the English text (simple matching)
        spans = []
        if en and kws:
            lowered = en.lower()
            for k in kws:
                k_low = k.lower()
                start = lowered.find(k_low)
                if start != -1:
                    spans.append((start, start + len(k), "keyword"))
        return det, en, summ, {"text": en, "spans": spans} if en else []

    run_btn.click(
        _ui_wrapper,
        inputs=[inp, max_len, topk],
        outputs=[detected_lang, english_out, summary_out, keywords_out],
    )

    gr.Examples(
        examples=EXAMPLE_INPUTS,
        inputs=[inp],
        label="Try examples",
        examples_per_page=3,
    )

if __name__ == "__main__":
    # For local dev, open in browser on http://127.0.0.1:7860
    demo.launch()
