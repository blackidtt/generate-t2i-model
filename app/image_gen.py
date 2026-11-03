"""Generator gambar. Mode asli memakai SDXL-Turbo, mode MOCK menggambar kartu warna dari hash prompt."""
import hashlib
import textwrap
import threading

from PIL import Image, ImageDraw

from . import config

_lock = threading.Lock()
_pipe = None


def _load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import AutoPipelineForText2Image

    if torch.cuda.is_available():
        device, dtype, variant = "cuda", torch.float16, "fp16"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype, variant = "mps", torch.float16, None
    else:
        device, dtype, variant = "cpu", torch.float32, None
    kwargs = {"torch_dtype": dtype}
    if variant:
        kwargs["variant"] = variant
    pipe = AutoPipelineForText2Image.from_pretrained(config.SDXL_MODEL, **kwargs)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    return _pipe


def _mock_image(prompt: str) -> Image.Image:
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    c1 = (h[0], h[1], h[2])
    c2 = (h[3], h[4], h[5])
    size = config.IMG_SIZE
    img = Image.new("RGB", (size, size), c1)
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        row = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        for x in range(size):
            px[x, y] = row
    draw = ImageDraw.Draw(img)
    text = "\n".join(textwrap.wrap(prompt, 34)[:12])
    draw.rectangle([20, 20, size - 20, size - 20], outline=(255, 255, 255), width=2)
    draw.text((32, 32), "MOCK IMAGE\n\n" + text, fill=(255, 255, 255))
    return img


def generate(prompt: str) -> Image.Image:
    """Menghasilkan satu gambar. Seed tetap sehingga hanya prompt yang memengaruhi hasil."""
    if config.MOCK:
        return _mock_image(prompt)
    import torch

    with _lock:
        pipe = _load_pipe()
        generator = torch.Generator("cpu").manual_seed(config.SEED)
        out = pipe(
            prompt=prompt,
            num_inference_steps=1,   # SDXL-Turbo dirancang untuk 1 sampai 4 langkah
            guidance_scale=0.0,      # SDXL-Turbo dilatih tanpa classifier-free guidance
            height=config.IMG_SIZE,
            width=config.IMG_SIZE,
            generator=generator,
        )
    return out.images[0]
