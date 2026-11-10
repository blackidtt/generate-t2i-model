"""Logika sesi: counterbalancing, tahapan, iterasi, skor, dan pencatatan log."""
import base64
import difflib
import io
import json
import random
import re
import threading
import time

from PIL import Image

from . import config, image_gen, lang, llm, similarity

IPE, CFU = "IPE", "CFU"
ORDERS = {"AB": [IPE, CFU], "BA": [CFU, IPE]}   # A = In-Place Editing, B = Conversational Follow-Up
STAGES = ["consent", "guide", "task_1", "break", "task_2", "done"]
PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

_io_lock = threading.RLock()


class AppError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# ---------- penyimpanan ----------
def _sess_path(pid):
    return config.SESSIONS_DIR / f"{pid}.json"


def _check_pid(pid: str) -> str:
    if not PID_RE.match(pid or ""):
        raise AppError("Kode partisipan hanya boleh huruf, angka, garis bawah, atau strip (maks 32 karakter).")
    return pid


def load(pid: str) -> dict:
    p = _sess_path(_check_pid(pid))
    if not p.exists():
        raise AppError("Sesi tidak ditemukan. Daftarkan kode partisipan terlebih dahulu.", 404)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("stage") in ("practice_1", "practice_2"):
        data["stage"] = "guide"
    return data


def save(sess: dict) -> None:
    with _io_lock:
        _sess_path(sess["pid"]).write_text(json.dumps(sess, ensure_ascii=False, indent=2), encoding="utf-8")


def log(event: str, pid: str, **fields) -> None:
    rec = {"ts": round(time.time(), 3), "event": event, "pid": pid, **fields}
    with _io_lock:
        with config.LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------- pendaftaran dan counterbalancing ----------
def _assign_order() -> str:
    counts = {"AB": 0, "BA": 0}
    for p in config.SESSIONS_DIR.glob("*.json"):
        try:
            counts[json.loads(p.read_text(encoding="utf-8"))["order"]] += 1
        except Exception:  # noqa: BLE001
            pass
    if counts["AB"] == counts["BA"]:
        return random.choice(["AB", "BA"])
    return "AB" if counts["AB"] < counts["BA"] else "BA"


def register(pid: str, order: str | None = None) -> dict:
    with _io_lock:
        _check_pid(pid)
        if _sess_path(pid).exists():
            return public_state(load(pid))
        if order is not None and order not in ORDERS:
            raise AppError("Urutan harus AB atau BA.")
        sess = {
            "pid": pid,
            "order": order or _assign_order(),
            "stage": "consent",
            "created": time.time(),
            "practice": {IPE: {"prompt": "", "n": 0}, CFU: {"prompt": "", "n": 0}},
            "tasks": [],
        }
        save(sess)
        log("register", pid, order=sess["order"])
        return public_state(sess)


def public_state(sess: dict) -> dict:
    """Status yang aman dikirim ke browser (tanpa skor dan tanpa prompt rahasia)."""
    stage = sess["stage"]
    mode = None
    if stage in ("task_1", "task_2"):
        mode = ORDERS[sess["order"]][int(stage[-1]) - 1]
    out = {"pid": sess["pid"], "stage": stage, "mode": mode, "order": sess.get("order")}
    if stage in ("task_1", "task_2"):
        idx = int(stage[-1])
        task = next((t for t in sess["tasks"] if t["index"] == idx), None)
        out["task_started"] = task is not None
        out["task_done"] = bool(task and task["done"])
        out["iterations_so_far"] = task["n_iter"] if task else 0
    return out


def advance(pid: str, from_stage: str) -> dict:
    with _io_lock:
        sess = load(pid)
        if from_stage in ("practice_1", "practice_2"):
            from_stage = "guide"
        if sess["stage"] != from_stage:
            raise AppError("Tahap tidak sesuai. Muat ulang halaman.")
        if from_stage in ("task_1", "task_2"):
            task = next((t for t in sess["tasks"] if t["index"] == int(from_stage[-1])), None)
            if not task or not task["done"]:
                raise AppError("Tugas belum selesai.")
        i = STAGES.index(from_stage)
        if i + 1 >= len(STAGES):
            raise AppError("Sesi sudah selesai.")
        sess["stage"] = STAGES[i + 1]
        save(sess)
        log("advance", pid, from_stage=from_stage, to_stage=sess["stage"])
        return public_state(sess)


# ---------- utilitas iterasi ----------
def _words(text: str):
    return re.findall(r"\S+", text.strip())


def _diff(old: str, new: str):
    a, b = _words(old.lower()), _words(new.lower())
    added, removed = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag in ("replace", "delete"):
            removed += a[i1:i2]
        if tag in ("replace", "insert"):
            added += b[j1:j2]
    return added, removed


def _to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _produce(mode: str, current_prompt: str, n_iter: int, text: str):
    """Menentukan prompt baru dari masukan partisipan. Mengembalikan (prompt_baru, kata_diketik, llm_meta, added, removed)."""
    text = (text or "").strip()
    if not text:
        raise AppError("Masukan tidak boleh kosong.")
    llm_meta = None
    if mode == IPE:
        new_prompt = text
        if n_iter > 0 and new_prompt == current_prompt:
            raise AppError("Prompt belum diubah. Ubah prompt sebelum menekan Generate.")
        added, removed = _diff(current_prompt, new_prompt) if n_iter > 0 else (_words(new_prompt), [])
        typed = len(added) if n_iter > 0 else len(_words(new_prompt))
    else:
        if n_iter == 0:
            new_prompt = text
            added, removed = _words(text), []
        else:
            new_prompt, llm_meta = llm.revise(current_prompt, text)
            added, removed = _diff(current_prompt, new_prompt)
        typed = len(_words(text))
    return new_prompt, typed, llm_meta, added, removed


# ---------- latihan ----------
def practice_step(pid: str, text: str) -> dict:
    with _io_lock:
        sess = load(pid)
        if sess["stage"] not in ("practice_1", "practice_2"):
            raise AppError("Bukan tahap latihan.")
        mode = ORDERS[sess["order"]][int(sess["stage"][-1]) - 1]
        pr = sess["practice"][mode]
    new_prompt, typed, llm_meta, _, _ = _produce(mode, pr["prompt"], pr["n"], text)
    img = image_gen.generate(new_prompt)
    flag = lang.non_english_suspected(text)
    with _io_lock:
        sess = load(pid)
        sess["practice"][mode] = {"prompt": new_prompt, "n": pr["n"] + 1}
        save(sess)
        log("practice_step", pid, mode=mode, n=pr["n"] + 1, user_input=text, prompt_after=new_prompt,
            non_english_suspected=flag, llm_error=(llm_meta or {}).get("error"))
    return {"image": _to_data_uri(img), "non_english": flag, "n": pr["n"] + 1,
            "prompt": new_prompt if (mode == IPE or config.SHOW_LLM_PROMPT) else None}


# ---------- tugas utama ----------
def _current_task(sess: dict):
    if sess["stage"] not in ("task_1", "task_2"):
        raise AppError("Bukan tahap tugas.")
    idx = int(sess["stage"][-1])
    task = next((t for t in sess["tasks"] if t["index"] == idx), None)
    return idx, task


def start_task(pid: str) -> dict:
    with _io_lock:
        sess = load(pid)
        idx, task = _current_task(sess)
        cond = ORDERS[sess["order"]][idx - 1]
        if task is None:
            task = {"index": idx, "condition": cond, "started_at": time.time(), "current_prompt": "",
                    "n_iter": 0, "done": False, "completed": False, "iters_to_threshold": None,
                    "best_score": 0.0, "words_typed": 0}
            sess["tasks"].append(task)
            save(sess)
            log("task_start", pid, task=idx, condition=cond, order=sess["order"])
        last_image, shown_prompt = None, None
        files = sorted((config.IMAGES_DIR / pid).glob(f"task{idx}_{cond}_iter*.png"))
        if files:
            last_image = _to_data_uri(Image.open(files[-1]))
            if cond == IPE or config.SHOW_LLM_PROMPT:
                shown_prompt = task["current_prompt"]
        return {"condition": cond, "task": idx, "done": task["done"], "target_url": "/static/target.png",
                "n_iter": task["n_iter"], "last_image": last_image, "prompt": shown_prompt}


def task_step(pid: str, text: str) -> dict:
    with _io_lock:
        sess = load(pid)
        idx, task = _current_task(sess)
        if task is None:
            raise AppError("Tugas belum dimulai.")
        if task["done"]:
            raise AppError("Tugas sudah selesai.")
        cond, cur, n_iter = task["condition"], task["current_prompt"], task["n_iter"]
    t0 = time.time()
    new_prompt, typed, llm_meta, added, removed = _produce(cond, cur, n_iter, text)
    img = image_gen.generate(new_prompt)
    sc = similarity.score(img, new_prompt)
    flag = lang.non_english_suspected(text)
    with _io_lock:
        sess = load(pid)
        idx, task = _current_task(sess)
        k = task["n_iter"] + 1
        elapsed = round(time.time() - task["started_at"], 3)
        reached = sc >= config.THRESHOLD
        task.update(current_prompt=new_prompt, n_iter=k, words_typed=task["words_typed"] + typed,
                    best_score=max(task["best_score"], sc))
        img_dir = config.IMAGES_DIR / pid
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(img_dir / f"task{idx}_{cond}_iter{k:03d}.png")
        log("iteration", pid, task=idx, condition=cond, iteration=k, user_input=text, prompt_before=cur,
            prompt_after=new_prompt, added_words=added, removed_words=removed, words_typed=typed,
            score=round(sc, 5), reached=reached, elapsed_s=elapsed, gen_s=round(time.time() - t0, 3),
            non_english_suspected=flag, llm_error=(llm_meta or {}).get("error"),
            llm_unchanged=bool(llm_meta and not llm_meta.get("error") and new_prompt == cur))
        if reached:
            task.update(done=True, completed=True, iters_to_threshold=k)
            log("task_end", pid, task=idx, condition=cond, completed=True, iterations=k,
                best_score=round(task["best_score"], 5), elapsed_s=elapsed, words_typed=task["words_typed"])
        save(sess)
    show = new_prompt if (cond == IPE or config.SHOW_LLM_PROMPT) else None
    return {"image": _to_data_uri(img), "done": task["done"], "completed": task["completed"],
            "non_english": flag, "prompt": show, "llm_error": bool(llm_meta and llm_meta.get("error")),
            "score": round(sc, 4) if config.SHOW_SCORE else None, "gen_s": round(time.time() - t0, 2)}


def give_up(pid: str) -> dict:
    with _io_lock:
        sess = load(pid)
        idx, task = _current_task(sess)
        if task is None or task["done"]:
            raise AppError("Tidak ada tugas aktif.")
        task.update(done=True, completed=False)
        elapsed = round(time.time() - task["started_at"], 3)
        log("task_end", pid, task=idx, condition=task["condition"], completed=False, iterations=task["n_iter"],
            best_score=round(task["best_score"], 5), elapsed_s=elapsed, words_typed=task["words_typed"])
        save(sess)
        return {"done": True, "completed": False}
