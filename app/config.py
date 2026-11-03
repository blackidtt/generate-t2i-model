"""Konfigurasi aplikasi demo. Semua nilai bisa diganti lewat environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
STATIC_DIR = BASE_DIR / "static"
SESSIONS_DIR = DATA_DIR / "sessions"
IMAGES_DIR = DATA_DIR / "images"
LOG_FILE = DATA_DIR / "logs" / "events.jsonl"
TARGET_PROMPT_FILE = DATA_DIR / "secret" / "target_prompt.txt"
TARGET_IMAGE = STATIC_DIR / "target.png"

# MOCK=1 memakai generator dan skor tiruan sehingga UI bisa dicoba tanpa GPU, SDXL-Turbo, Ollama, atau CLIP
MOCK = os.getenv("MOCK", "0") == "1"

# Parameter penelitian (proposal bagian 5.4 dan 5.5)
THRESHOLD = float(os.getenv("THRESHOLD", "0.85"))   # ambang kemiripan CLIP
SEED = int(os.getenv("SEED", "42"))                  # seed gambar tetap di semua iterasi
IMG_SIZE = 512

# Model gambar
SDXL_MODEL = os.getenv("SDXL_MODEL", "stabilityai/sdxl-turbo")

# Model bahasa penerjemah untuk mode Conversational Follow-Up
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# Metrik kemiripan
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Prompt hasil terjemahan LLM tidak ditampilkan ke partisipan (tetap dicatat di log)
SHOW_LLM_PROMPT = os.getenv("SHOW_LLM_PROMPT", "0") == "1"

# Hanya untuk pilot atau debugging oleh peneliti. WAJIB 0 saat mengumpulkan data penelitian
SHOW_SCORE = os.getenv("SHOW_SCORE", "0") == "1"

# Memuat model ke memori saat server menyala agar generate pertama tidak lambat
WARMUP = os.getenv("WARMUP", "1") == "1"
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "60m")   # jaga Qwen3 tetap di memori


def read_target_prompt() -> str:
    """Prompt rahasia gambar target. Dipakai skrip make_target.py dan mode MOCK."""
    env = os.getenv("TARGET_PROMPT")
    if env:
        return env.strip()
    if TARGET_PROMPT_FILE.exists():
        return TARGET_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return "a red fox sitting in a snowy forest, golden sunrise light, detailed fur"


for d in (SESSIONS_DIR, IMAGES_DIR, LOG_FILE.parent, TARGET_PROMPT_FILE.parent):
    d.mkdir(parents=True, exist_ok=True)
