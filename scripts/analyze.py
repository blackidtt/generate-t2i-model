"""Analisis data sesuai proposal bagian 5.7.

RQ1  : jumlah iterasi sampai ambang tercapai, IPE vs CFU (paired t-test atau Wilcoxon), ukuran efek dz,
       CI 95 persen, pengecekan efek urutan, dan DV pendukung (skor terbaik, waktu, jumlah kata).
RQ2  : ekspor revisi untuk dikode (--export-coding), lalu kappa dan uji per kategori (--coded).

Pemakaian:
  python -m scripts.analyze                              # RQ1 dan DV pendukung
  python -m scripts.analyze --export-coding coding.csv   # tabel revisi untuk dua pengkode
  python -m scripts.analyze --coded coding_filled.csv    # kappa dan perbandingan kategori RQ2
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from app import config

CATEGORIES = ["Adding descriptors", "Adding themes", "Adjusting view", "Omitting words",
              "Reordering or Rephrasing", "Lainnya"]


def load_events() -> pd.DataFrame:
    lines = [json.loads(l) for l in config.LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(lines)


def paired_test(d: np.ndarray, label: str):
    """Uji berpasangan. Normalitas selisih diperiksa dengan Shapiro-Wilk (alpha 0,05)."""
    n = len(d)
    out = {"n_pairs": n, "mean_diff": float(np.mean(d)), "sd_diff": float(np.std(d, ddof=1)) if n > 1 else float("nan")}
    if n < 3:
        print(f"[{label}] pasangan terlalu sedikit untuk uji (n={n})")
        return out
    p_norm = stats.shapiro(d).pvalue if np.ptp(d) > 0 else 0.0
    out["shapiro_p"] = float(p_norm)
    if p_norm > 0.05:
        t = stats.ttest_1samp(d, 0.0)
        se = out["sd_diff"] / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        out.update(test="paired t-test", stat=float(t.statistic), p=float(t.pvalue),
                   ci95=(out["mean_diff"] - h, out["mean_diff"] + h))
    else:
        w = stats.wilcoxon(d, zero_method="wilcox") if np.any(d != 0) else None
        out.update(test="Wilcoxon signed-rank", stat=float(w.statistic) if w else float("nan"),
                   p=float(w.pvalue) if w else 1.0, ci95=None)
    out["dz"] = out["mean_diff"] / out["sd_diff"] if out["sd_diff"] else float("nan")
    print(f"[{label}] {out['test']}: selisih rata-rata={out['mean_diff']:.3f}, dz={out['dz']:.3f}, "
          f"p={out['p']:.4f}, Shapiro p={p_norm:.3f}, "
          f"CI95={tuple(round(float(x), 3) for x in out['ci95']) if out['ci95'] else 'n/a'}")
    return out


def rq1(ev: pd.DataFrame):
    ends = ev[ev["event"] == "task_end"].copy()
    ends["completed"] = ends["completed"].astype(bool)
    reg = ev[ev["event"] == "register"].set_index("pid")["order"]
    print(f"Partisipan terdaftar: {len(reg)} | tugas selesai: {int(ends['completed'].sum())} | berhenti: {int((~ends['completed']).sum())}")
    wide = {}
    for col in ["iterations", "best_score", "elapsed_s", "words_typed"]:
        piv = ends[ends["completed"]].pivot_table(index="pid", columns="condition", values=col, aggfunc="first")
        wide[col] = piv.dropna()
    base = wide["iterations"]
    if base.empty:
        print("Belum ada partisipan yang menyelesaikan kedua tugas.")
        return
    print("\n=== RQ1: jumlah iterasi sampai ambang ===")
    print(base.describe().round(2).to_string())
    d = (base["IPE"] - base["CFU"]).to_numpy()   # positif = CFU butuh iterasi lebih sedikit (sesuai H1)
    paired_test(d, "RQ1 IPE minus CFU")
    print("\n=== Efek urutan (selisih IPE minus CFU menurut urutan) ===")
    od = pd.DataFrame({"d": base["IPE"] - base["CFU"], "order": reg.reindex(base.index)})
    g = [x["d"].to_numpy() for _, x in od.groupby("order")]
    if len(g) == 2 and min(len(x) for x in g) >= 3:
        normal = all(stats.shapiro(x).pvalue > 0.05 for x in g if np.ptp(x) > 0)
        res = stats.ttest_ind(*g, equal_var=False) if normal else stats.mannwhitneyu(*g)
        print(f"{'Welch t-test' if normal else 'Mann-Whitney'} p={res.pvalue:.4f} | rata-rata menurut urutan: "
              f"{od.groupby('order')['d'].mean().round(2).to_dict()}")
    else:
        print("Data per urutan belum cukup.")
    print("\n=== DV pendukung (IPE minus CFU) ===")
    for col, name in [("best_score", "skor terbaik"), ("elapsed_s", "waktu (detik)"), ("words_typed", "jumlah kata")]:
        w = wide[col]
        if not w.empty:
            paired_test((w["IPE"] - w["CFU"]).to_numpy(), name)
    out = base.copy()
    out.to_csv(config.DATA_DIR / "rq1_iterations.csv")
    print(f"\nTabel pasangan disimpan di {config.DATA_DIR / 'rq1_iterations.csv'}")


def export_coding(ev: pd.DataFrame, path: str):
    it = ev[(ev["event"] == "iteration") & (ev["iteration"] >= 2)].copy()
    cols = ["pid", "task", "condition", "iteration", "user_input", "prompt_before", "prompt_after",
            "added_words", "removed_words"]
    out = it[cols].copy()
    out["added_words"] = out["added_words"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    out["removed_words"] = out["removed_words"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    for c in ("code_coder1", "code_coder2", "code_final"):
        out[c] = ""
    out.to_csv(path, index=False)
    print(f"{len(out)} revisi diekspor ke {path}. Isi code_coder1 dan code_coder2 dengan salah satu dari: {CATEGORIES}")
    print("Kode dari input partisipan (user_input), bukan dari prompt keluaran LLM. Isi code_final setelah diskusi.")


def rq2(path: str):
    from sklearn.metrics import cohen_kappa_score
    from statsmodels.stats.multitest import multipletests

    df = pd.read_csv(path).fillna("")
    both = df[(df["code_coder1"] != "") & (df["code_coder2"] != "")]
    if len(both) >= 2:
        k = cohen_kappa_score(both["code_coder1"], both["code_coder2"])
        print(f"Cohen's kappa = {k:.3f} pada {len(both)} revisi (target minimal 0,70). "
              f"{'MEMENUHI' if k >= 0.70 else 'DI BAWAH TARGET, perbaiki codebook dan kode ulang'}")
    final = df[df["code_final"] != ""]
    if final.empty:
        print("code_final belum terisi, perbandingan kategori dilewati.")
        return
    cnt = final.groupby(["pid", "condition", "code_final"]).size().unstack("condition", fill_value=0)
    rows, pvals = [], []
    for cat in CATEGORIES:
        if cat not in cnt.index.get_level_values("code_final"):
            continue
        sub = cnt.xs(cat, level="code_final")
        for c in ("IPE", "CFU"):
            if c not in sub.columns:
                sub[c] = 0
        sub = sub.reindex(sorted(final["pid"].unique()), fill_value=0)
        d = (sub["IPE"] - sub["CFU"]).to_numpy()
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append((cat, int(sub["IPE"].sum()), int(sub["CFU"].sum()), p))
        pvals.append(p)
    if rows:
        adj = multipletests(pvals, method="holm")[1]
        print("\n=== RQ2: jumlah revisi per kategori (Wilcoxon, koreksi Holm) ===")
        for (cat, a, b, p), pa in zip(rows, adj):
            print(f"{cat:28s} IPE={a:4d} CFU={b:4d} p={p:.4f} p_holm={pa:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-coding")
    ap.add_argument("--coded")
    a = ap.parse_args()
    if a.coded:
        rq2(a.coded)
    elif a.export_coding:
        export_coding(load_events(), a.export_coding)
    else:
        rq1(load_events())
