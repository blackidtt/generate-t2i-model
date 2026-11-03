"""Skor kemiripan gambar ke gambar. Mode asli: kosinus embedding CLIP ViT-B/32. Mode MOCK: tumpang tindih kata."""
import re
import threading

from PIL import Image

from . import config

_lock = threading.Lock()
_state = {}


def _load_clip():
    if "model" in _state:
        return _state
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
    )
    model.eval().to(device)
    _state.update(model=model, preprocess=preprocess, device=device)
    target = Image.open(config.TARGET_IMAGE).convert("RGB")
    _state["target_emb"] = _embed(target)
    return _state


def _embed(img: Image.Image):
    import torch

    s = _state
    with torch.no_grad():
        x = s["preprocess"](img.convert("RGB")).unsqueeze(0).to(s["device"])
        f = s["model"].encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
    return f


def _words(text: str):
    return set(re.findall(r"[a-z]+", text.lower()))


def score(image: Image.Image, prompt: str) -> float:
    """Mengembalikan skor kemiripan dengan gambar target (0 sampai 1)."""
    if config.MOCK:
        a, b = _words(prompt), _words(config.read_target_prompt())
        if not a or not b:
            return 0.0
        f1 = 2 * len(a & b) / (len(a) + len(b))
        return round(0.6 + 0.4 * f1, 4)
    with _lock:
        s = _load_clip()
        emb = _embed(image)
        return float((emb @ s["target_emb"].T).item())
