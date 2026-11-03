"""Membantu memeriksa ambang kemiripan sebelum data utama dikumpulkan (pilot 3 sampai 5 orang).

Membaca log pilot dan mencetak, untuk tiap partisipan pilot dan kondisi, skor per iterasi, kapan ambang
tercapai, dan apakah ambang sudah tercapai pada iterasi pertama (tanda ambang terlalu longgar).
Prinsip proposal: ambang hanya boleh disesuaikan satu kali sebelum data utama, lalu dikunci.

Pemakaian: python -m scripts.pilot_threshold P01 P02 P03
"""
import json
import sys
from collections import defaultdict

from app import config


def main(pids):
    rows = [json.loads(line) for line in config.LOG_FILE.read_text(encoding="utf-8").splitlines() if line]
    by = defaultdict(list)
    for r in rows:
        if r["event"] == "iteration" and (not pids or r["pid"] in pids):
            by[(r["pid"], r["condition"])].append(r)
    print(f"Ambang saat ini: {config.THRESHOLD}")
    for (pid, cond), its in sorted(by.items()):
        scores = [round(i["score"], 3) for i in sorted(its, key=lambda x: x["iteration"])]
        hit = next((i["iteration"] for i in its if i["reached"]), None)
        note = "  <-- tercapai di iterasi 1, ambang mungkin terlalu longgar" if hit == 1 else ""
        print(f"{pid} {cond}: skor={scores} tercapai_di={hit}{note}")


if __name__ == "__main__":
    main(set(sys.argv[1:]))
