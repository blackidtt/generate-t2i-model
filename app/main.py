"""Server FastAPI untuk aplikasi demo eksperimen IPE vs CFU.

Jalankan: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import config, core

app = FastAPI(title="Demo IPE vs CFU")


@app.on_event("startup")
def warmup():
    """Memuat Z-Image-Turbo, CLIP, dan Qwen3 ke memori sekali di awal, lalu mencetak waktunya."""
    if config.MOCK or not config.WARMUP:
        return
    import time
    from . import image_gen, llm, similarity

    t = time.time()
    img = image_gen.generate("a photo of a cat")
    print(f"[warmup] Z-Image-Turbo siap ({time.time() - t:.1f} detik pemuatan + generate pertama)")
    t = time.time()
    image_gen.generate("a photo of a dog")
    print(f"[warmup] generate berikutnya: {time.time() - t:.2f} detik per gambar")
    try:
        t = time.time()
        similarity.score(img, "a photo of a cat")
        print(f"[warmup] CLIP siap ({time.time() - t:.1f} detik)")
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] CLIP belum siap, kemungkinan static/target.png belum dibuat: {e}")
    t = time.time()
    _, meta = llm.revise("a photo of a cat", "make it black")
    print(f"[warmup] Qwen3 {'GAGAL: ' + str(meta['error']) if meta['error'] else 'siap'} ({time.time() - t:.1f} detik)")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.exception_handler(core.AppError)
async def app_error_handler(_: Request, exc: core.AppError):
    return JSONResponse({"error": exc.message}, status_code=exc.status)


class Register(BaseModel):
    pid: str
    order: str | None = None   # opsional: paksa urutan AB atau BA (hanya untuk peneliti)


class Pid(BaseModel):
    pid: str


class Advance(BaseModel):
    pid: str
    from_stage: str


class Step(BaseModel):
    pid: str
    text: str


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return {"mock": config.MOCK}


@app.post("/api/register")
def register(body: Register):
    return core.register(body.pid, body.order)


@app.get("/api/state")
def state(pid: str):
    return core.public_state(core.load(pid))


@app.post("/api/advance")
def advance(body: Advance):
    return core.advance(body.pid, body.from_stage)


@app.post("/api/practice/step")
async def practice_step(body: Step):
    return await run_in_threadpool(core.practice_step, body.pid, body.text)


@app.post("/api/task/start")
def task_start(body: Pid):
    return core.start_task(body.pid)


@app.post("/api/task/step")
async def task_step(body: Step):
    return await run_in_threadpool(core.task_step, body.pid, body.text)


@app.post("/api/task/giveup")
def task_giveup(body: Pid):
    return core.give_up(body.pid)
