"""Lapisan LLM untuk mode Conversational Follow-Up.

Satu LLM, satu instruksi sistem tetap, temperature 0, dan seed tetap (proposal bagian 5.4).
LLM hanya menerjemahkan instruksi lanjutan menjadi prompt baru. Ia tidak menilai gambar.
"""
import re

import requests

from . import config

SYSTEM_PROMPT = (
    "You rewrite text-to-image prompts. You receive a CURRENT PROMPT and a user's FOLLOW-UP INSTRUCTION.\n"
    "Rules:\n"
    "1. Apply only the change the instruction asks for. Keep every other word of the current prompt unchanged.\n"
    "2. Make the smallest edit possible. Do not add details the user did not ask for.\n"
    "3. Always output the new prompt in English, even if the instruction is in another language.\n"
    "4. Output only the new prompt on one line. No quotes, no labels, no explanations."
)


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.split("</think>")[-1]   # template Ollama bisa membuka <think> sendiri sehingga hanya penutupnya yang muncul
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"^(new prompt|prompt)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip("'").strip()


def revise(current_prompt: str, instruction: str):
    """Mengembalikan (prompt_baru, meta). Jika LLM gagal, prompt lama dipertahankan dan meta['error'] terisi."""
    meta = {"model": "mock" if config.MOCK else config.OLLAMA_MODEL, "error": None}
    if config.MOCK:
        return f"{current_prompt}, {instruction.strip()}", meta
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CURRENT PROMPT: {current_prompt}\nFOLLOW-UP INSTRUCTION: {instruction}",
            },
        ],
        "options": {"temperature": 0, "seed": 42, "num_predict": 160},
    }
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        new_prompt = _clean(r.json()["message"]["content"])
        if not new_prompt:
            raise ValueError("keluaran LLM kosong")
        return new_prompt, meta
    except Exception as e:  # noqa: BLE001
        meta["error"] = str(e)
        return current_prompt, meta
