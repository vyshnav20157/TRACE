"""Significance testing over the saved per-sample predictions.

Every `<run>_preds.json` in the repo root stores the test-set `labels` and `predictions`
for one (dataset, task, captioner, arm) run. Within a dataset all runs are scored on the
same rows in the same order (asserted at load), so two runs' predictions are PAIRED. That
pairing is what makes these tests far more powerful than comparing two headline numbers.

WHAT IS TESTED, AND WHY EACH TEST
---------------------------------
1. McNemar (exact) on ACCURACY.
   Accuracy decomposes per sample, so the paired disagreement counts are sufficient
   statistics:
       b01 = A correct, B wrong        b10 = A wrong, B correct
   Under H0 "on a discordant sample either model is equally likely to be the right one",
   b01 ~ Binomial(b01+b10, 0.5). The EXACT binomial test is used, not the chi-square
   approximation, because several arms here have few discordant pairs.

2. Paired bootstrap on MACRO-F1 (and binary-F1 / the positive-class F1).
   F1 is a ratio of sums and does NOT decompose per sample, so McNemar cannot test it.
   Resampling the shared rows and recomputing both models' F1 on the SAME resample keeps
   the pairing and yields both a CI and a two-sided p-value for the delta. This matters
   most on Memotion, which is 76%/62% positive -- there accuracy is a weak metric and
   macro-F1 is the headline, so testing only accuracy would miss the real effect.

3. Bootstrap CI for each run's own metrics (`--per-run`).
   Lets the thesis report "macro-F1 0.7396 [0.71, 0.77]" instead of a bare point estimate.

4. Effect sizes, reported alongside every p-value:
     - odds ratio b01/b10 on the discordant pairs (McNemar's natural effect size)
     - Cohen's g = |b01/(b01+b10) - 0.5|, the standard paired-proportion effect size
     - raw deltas in accuracy and F1
   A p-value says an effect is real; these say whether it is big enough to care about.

5. Cohen's kappa between the two runs' predictions -- how much two arms agree at all,
   which distinguishes "different models" from "same model, noisy".

6. Multiple-comparison control. All pairs within one dataset form one family, so raw
   p-values are adjusted with Holm-Bonferroni (strong FWER control, uniformly more
   powerful than plain Bonferroni). Benjamini-Hochberg FDR is reported too via `--fdr`,
   which is the more appropriate correction for a large exploratory grid.

NOT TESTED, AND WHY
-------------------
AUC and the tuned-threshold metrics cannot be tested here: the training scripts compute
`all_probs` but save only the thresholded `predictions` (clip_xlm_roberta_ft.py:599), so
the per-sample scores those metrics need are not on disk. Testing them would require
re-running eval with probabilities saved. Everything above needs only hard labels.

LOGGING
-------
Every run APPENDS its rows to `significance_results.jsonl` in the repo root by default,
matching the append-only convention of the per-dataset `evalresults.jsonl`. One JSON object
per line, `kind` = "comparison" or "per_run". Each row carries its own provenance --
timestamp, git commit, the exact command, seed, n_boot, the correction family, and a SHA1
digest of each source prediction vector -- so a result can be traced back to the predictions
that produced it, and a re-run after retraining an arm is distinguishable from the old row
rather than silently conflated with it. `--no-log` disables; `--out FILE` additionally
freezes one invocation as a single self-contained JSON snapshot.

Note the log is APPEND-only, so it accumulates across runs. Read the latest result for a
pair by taking the row with the newest `timestamp`.

USAGE
-----
    python utils/significance.py --dataset all --fdr --format markdown
    python utils/significance.py --dataset mmsd --baseline mmsd_roberta_trace --bootstrap
    python utils/significance.py --dataset all --per-run
    python utils/significance.py --dataset mmsd --no-log          # scratch run, nothing logged
"""

import argparse
import datetime
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which preds files belong to which comparison group. A run is only comparable to others
# scored on the same rows, so Memotion's two tasks are separate groups even though they
# happen to share a split size.
GROUPS = {
    "mami": "mami_*_preds.json",
    "memotion_humour": "memotion_humour_*_preds.json",
    "memotion_offensive": "memotion_offensive_*_preds.json",
    "mmsd": "mmsd_*_preds.json",
}


def run_name(path):
    return os.path.basename(path)[: -len("_preds.json")]


def load_group(pattern):
    """Load every run in one group, asserting all are scored on the same rows.

    Also returns per-run provenance: a digest of the prediction vector and the source
    file's mtime. A logged result is only meaningful if you can tell WHICH predictions
    produced it, so re-running an arm and re-testing yields a different digest and the two
    log rows stay distinguishable.
    """
    runs, labels, prov = {}, None, {}
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, pattern))):
        data = json.load(open(path))
        y = np.asarray(data["labels"], dtype=int)
        p = np.asarray(data["predictions"], dtype=int)
        if len(y) != len(p):
            raise SystemExit(f"{path}: {len(y)} labels vs {len(p)} predictions")
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            # Unpaired rows would invalidate every test below, so refuse rather than mislead.
            raise SystemExit(f"{path}: label vector differs from the rest of its group")
        name = run_name(path)
        runs[name] = p
        prov[name] = {
            "preds_file": os.path.relpath(path, REPO_ROOT),
            "preds_sha1": hashlib.sha1(p.tobytes()).hexdigest()[:12],
            "preds_mtime": datetime.datetime.fromtimestamp(
                os.path.getmtime(path)).isoformat(timespec="seconds"),
        }
    return labels, runs, prov


# ---------------------------------------------------------------------------
# Vectorized metrics. These are called ~10k times per pair under the bootstrap,
# so they use bincount rather than sklearn (which is ~50x slower here).
# ---------------------------------------------------------------------------

def _counts(y, p):
    """Return (tn, fp, fn, tp) for binary 0/1 arrays."""
    k = np.bincount(y * 2 + p, minlength=4)
    return k[0], k[1], k[2], k[3]


def macro_f1_fast(y, p):
    """Macro-F1, matching sklearn's f1_score(average="macro", zero_division=0).

    sklearn averages over the labels PRESENT in y_true/y_pred, not always over {0,1}. A
    bootstrap resample of an imbalanced split (Memotion is 76% positive) can legitimately
    draw only one class, and averaging a phantom 0.0 for the absent class there would drag
    the CI down. So the denominator counts only the classes actually present.
    """
    tn, fp, fn, tp = _counts(y, p)
    # 2tp/(2tp+fp+fn) is the F1 identity; the negative class is the same with tn for tp.
    pos_den = 2 * tp + fp + fn
    neg_den = 2 * tn + fn + fp
    pos = (2 * tp / pos_den) if pos_den else 0.0
    neg = (2 * tn / neg_den) if neg_den else 0.0
    # sklearn averages over the labels present in the UNION of y_true and y_pred. Both
    # labels appear unless the arrays are entirely one class, in which case it averages
    # over that single label. A bootstrap resample of an imbalanced split (Memotion is 76%
    # positive) can legitimately draw one class only, so this branch is reachable.
    single = (tn + fp + fn == 0) or (tp + fp + fn == 0)
    return (pos + neg) if single else (pos + neg) / 2


def binary_f1_fast(y, p):
    _, fp, fn, tp = _counts(y, p)
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else 0.0


def accuracy_fast(y, p):
    return float((y == p).mean())


METRICS = {
    "accuracy": accuracy_fast,
    "macro_f1": macro_f1_fast,
    "binary_f1": binary_f1_fast,
}


def mcnemar_exact(labels, pred_a, pred_b):
    """Exact McNemar on the discordant pairs, with its effect sizes."""
    ca, cb = pred_a == labels, pred_b == labels
    b01 = int(np.sum(ca & ~cb))
    b10 = int(np.sum(~ca & cb))
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    # Odds ratio of "A rescues it" vs "B rescues it"; inf when B never wins a discordant pair.
    odds = (b01 / b10) if b10 else (float("inf") if b01 else 1.0)
    # Cohen's g: departure of the discordant proportion from 0.5. 0.05/0.15/0.25 = S/M/L.
    g = abs(b01 / n - 0.5) if n else 0.0
    return {"b01": b01, "b10": b10, "n_discordant": n, "p_mcnemar": float(p),
            "odds_ratio": float(odds), "cohens_g": float(g),
            "acc_a": float(ca.mean()), "acc_b": float(cb.mean())}


def _boot_counts(labels, pred, idx):
    """Confusion counts for every bootstrap resample at once.

    `idx` is (n_boot, n) of row indices. Encoding each row as 2*y+p turns the per-resample
    confusion matrix into a bincount, which numpy does for all resamples in one pass -- the
    difference between ~14s and ~0.05s per pair, which is what makes the full grid runnable.
    """
    code = (labels * 2 + pred)[idx]                     # (n_boot, n) in {0,1,2,3}
    offset = code + 4 * np.arange(idx.shape[0])[:, None]
    flat = np.bincount(offset.ravel(), minlength=4 * idx.shape[0])
    return flat.reshape(idx.shape[0], 4)                # columns: tn, fp, fn, tp


def _metric_from_counts(name, c):
    """Vectorized metric over an (n_boot, 4) count matrix; mirrors the scalar versions."""
    tn, fp, fn, tp = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    total = tn + fp + fn + tp
    if name == "accuracy":
        return (tn + tp) / np.maximum(total, 1)
    pos_den = 2 * tp + fp + fn
    pos = np.divide(2 * tp, pos_den, out=np.zeros(len(c)), where=pos_den > 0)
    if name == "binary_f1":
        return pos
    neg_den = 2 * tn + fn + fp
    neg = np.divide(2 * tn, neg_den, out=np.zeros(len(c)), where=neg_den > 0)
    # Same single-label rule as macro_f1_fast.
    single = (tn + fp + fn == 0) | (tp + fp + fn == 0)
    return np.where(single, pos + neg, (pos + neg) / 2)


def paired_bootstrap(labels, pred_a, pred_b, metrics, n_boot, seed, block=2000):
    """Paired bootstrap over shared rows: CI and two-sided p for each metric's delta.

    Both models are scored on the SAME resampled rows every iteration, which preserves the
    pairing and removes the between-sample variance an unpaired bootstrap would leave in.
    Resamples are drawn in blocks so the index matrix never exceeds a few hundred MB.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = {m: np.empty(n_boot) for m in metrics}
    done = 0
    while done < n_boot:
        k = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        ca, cb = _boot_counts(labels, pred_a, idx), _boot_counts(labels, pred_b, idx)
        for m in metrics:
            deltas[m][done:done + k] = _metric_from_counts(m, ca) - _metric_from_counts(m, cb)
        done += k

    out = {}
    for m in metrics:
        obs = METRICS[m](labels, pred_a) - METRICS[m](labels, pred_b)
        d = deltas[m]
        # Two-sided bootstrap p: how often the resampled delta crosses zero relative to the
        # observed direction. +1 keeps p > 0 (never claim p == 0 from finite resamples).
        tail = np.mean(d <= 0) if obs > 0 else np.mean(d >= 0)
        p = min(1.0, 2 * (tail * n_boot + 1) / (n_boot + 1))
        out[m] = {"delta": float(obs),
                  "ci_low": float(np.percentile(d, 2.5)),
                  "ci_high": float(np.percentile(d, 97.5)),
                  "p_boot": float(p)}
    return out


def bootstrap_point(labels, pred, metrics, n_boot, seed, block=2000):
    """Bootstrap CI for one run's own metrics (same vectorized machinery)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = {m: np.empty(n_boot) for m in metrics}
    done = 0
    while done < n_boot:
        k = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(k, n))
        c = _boot_counts(labels, pred, idx)
        for m in metrics:
            vals[m][done:done + k] = _metric_from_counts(m, c)
        done += k
    return {m: {"value": float(METRICS[m](labels, pred)),
                "ci_low": float(np.percentile(vals[m], 2.5)),
                "ci_high": float(np.percentile(vals[m], 97.5))} for m in metrics}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (strong FWER control)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals):
    """BH adjusted p-values (FDR control) -- the better fit for a large exploratory grid."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m * pvals[i] / (rank + 1))
        adj[i] = min(1.0, running)
    return adj


def run_provenance(args):
    """Metadata identifying this invocation, stamped on every logged row."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit or None,
        "command": "python " + " ".join([os.path.relpath(sys.argv[0], REPO_ROOT)] + sys.argv[1:]),
        "n_boot": args.n_boot if args.bootstrap or args.per_run else None,
        "seed": args.seed,
        "alpha": args.alpha,
        "correction_family": "baseline" if args.baseline else "all_pairs",
        "baseline": args.baseline,
    }


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def effect_label(g):
    """Cohen's conventional bands for g."""
    return "large" if g >= 0.25 else "medium" if g >= 0.15 else "small" if g >= 0.05 else "negligible"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=list(GROUPS) + ["all"])
    ap.add_argument("--baseline", default=None,
                    help="Compare every run against this one run instead of all pairs. "
                         "Shrinks the correction family, so it is the powerful option.")
    ap.add_argument("--bootstrap", action="store_true",
                    help="Paired bootstrap p-values and CIs for the F1 metrics.")
    ap.add_argument("--per-run", action="store_true",
                    help="Also bootstrap a CI for each run's own metrics.")
    ap.add_argument("--metrics", default="accuracy,macro_f1",
                    help="Comma-separated subset of: " + ",".join(METRICS))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fdr", action="store_true",
                    help="Also report Benjamini-Hochberg FDR-adjusted p-values.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--format", default="text", choices=["text", "markdown", "csv"])
    ap.add_argument("--log-file", dest="log_file",
                    default=os.path.join(REPO_ROOT, "significance_results.jsonl"),
                    help="Append every comparison here as JSON lines (default: "
                         "significance_results.jsonl in the repo root). Matches the "
                         "append-only convention of the per-dataset evalresults.jsonl.")
    ap.add_argument("--no-log", action="store_true",
                    help="Do not append to the log file.")
    ap.add_argument("--out", default=None,
                    help="ALSO write this run's results as one self-contained JSON "
                         "snapshot (overwrites). Use for a table you want to freeze.")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise SystemExit(f"unknown metric(s): {bad}. choose from {list(METRICS)}")

    names = list(GROUPS) if args.dataset == "all" else [args.dataset]
    records, per_run_records = [], []
    meta = run_provenance(args)

    for group in names:
        labels, runs, prov = load_group(GROUPS[group])
        if len(runs) < 2:
            continue

        if args.per_run:
            print(f"\n### {group} -- per-run metrics, 95% bootstrap CI (n={len(labels)})")
            for name, pred in sorted(runs.items()):
                pt = bootstrap_point(labels, pred, metrics, args.n_boot, args.seed)
                per_run_records.append({"kind": "per_run", "group": group, "run": name,
                                        "n": int(len(labels)), **prov[name],
                                        "metrics": pt, **meta})
                cells = "  ".join(
                    f"{m}={pt[m]['value']:.4f} [{pt[m]['ci_low']:.4f},{pt[m]['ci_high']:.4f}]"
                    for m in metrics)
                print(f"  {name:<48} {cells}")

        if args.baseline:
            if args.baseline not in runs:
                raise SystemExit(f"--baseline {args.baseline} not in group {group}. "
                                 f"available: {sorted(runs)}")
            pairs = [(a, args.baseline) for a in runs if a != args.baseline]
        else:
            pairs = list(itertools.combinations(runs, 2))

        rows = []
        for a, b in pairs:
            r = {"kind": "comparison", "group": group, "a": a, "b": b,
                 "n": int(len(labels)),
                 "a_preds_sha1": prov[a]["preds_sha1"],
                 "b_preds_sha1": prov[b]["preds_sha1"]}
            r.update(mcnemar_exact(labels, runs[a], runs[b]))
            r["acc_delta"] = r["acc_a"] - r["acc_b"]
            r["kappa"] = float(cohen_kappa_score(runs[a], runs[b]))
            r["effect"] = effect_label(r["cohens_g"])
            for m in metrics:
                r[f"{m}_a"] = float(METRICS[m](labels, runs[a]))
                r[f"{m}_b"] = float(METRICS[m](labels, runs[b]))
                r[f"{m}_delta"] = r[f"{m}_a"] - r[f"{m}_b"]
            if args.bootstrap:
                bs = paired_bootstrap(labels, runs[a], runs[b], metrics,
                                      args.n_boot, args.seed)
                for m, v in bs.items():
                    r[f"{m}_ci_low"] = v["ci_low"]
                    r[f"{m}_ci_high"] = v["ci_high"]
                    r[f"{m}_p_boot"] = v["p_boot"]
            r.update(meta)
            rows.append(r)

        praw = [r["p_mcnemar"] for r in rows]
        for r, h in zip(rows, holm(praw)):
            r["p_holm"] = float(h)
            r["sig"] = stars(h)
        if args.fdr:
            for r, f in zip(rows, benjamini_hochberg(praw)):
                r["p_fdr"] = float(f)
        # The F1 bootstrap is its own family of tests, so it gets its own correction.
        if args.bootstrap:
            for m in metrics:
                pb = [r[f"{m}_p_boot"] for r in rows]
                for r, h in zip(rows, holm(pb)):
                    r[f"{m}_p_boot_holm"] = float(h)
        records.extend(rows)

        n_sig = sum(r["p_holm"] < args.alpha for r in rows)
        print(f"\n### {group}  (n={len(labels)}, {len(pairs)} comparisons, "
              f"{n_sig} significant at Holm alpha={args.alpha})")

        if args.format == "markdown":
            cols = ["A", "B", "Δacc", "A-only", "B-only", "OR", "g", "effect", "κ",
                    "p", "p(Holm)"]
            if args.fdr:
                cols.append("p(FDR)")
            for m in metrics:
                if m != "accuracy":
                    cols.append(f"Δ{m}")
                    if args.bootstrap:
                        cols.append(f"Δ{m} 95% CI")
                        cols.append(f"p({m})")
            cols.append("")
            print("| " + " | ".join(cols) + " |")
            print("|" + "---|" * len(cols))
            for r in sorted(rows, key=lambda r: r["p_holm"]):
                cells = [r["a"], r["b"], f"{r['acc_delta']:+.4f}",
                         str(r["b01"]), str(r["b10"]),
                         ("inf" if r["odds_ratio"] == float("inf") else f"{r['odds_ratio']:.2f}"),
                         f"{r['cohens_g']:.3f}", r["effect"], f"{r['kappa']:.3f}",
                         f"{r['p_mcnemar']:.3g}", f"{r['p_holm']:.3g}"]
                if args.fdr:
                    cells.append(f"{r['p_fdr']:.3g}")
                for m in metrics:
                    if m != "accuracy":
                        cells.append(f"{r[f'{m}_delta']:+.4f}")
                        if args.bootstrap:
                            cells.append(f"[{r[f'{m}_ci_low']:+.4f}, {r[f'{m}_ci_high']:+.4f}]")
                            cells.append(f"{r[f'{m}_p_boot']:.3g}")
                cells.append(r["sig"])
                print("| " + " | ".join(cells) + " |")

        elif args.format == "csv":
            keys = sorted({k for r in rows for k in r})
            print(",".join(keys))
            for r in sorted(rows, key=lambda r: r["p_holm"]):
                print(",".join(str(r.get(k, "")) for k in keys))

        else:
            for r in sorted(rows, key=lambda r: r["p_holm"]):
                extra = ""
                for m in metrics:
                    if m == "accuracy":
                        continue
                    extra += f" d{m}={r[f'{m}_delta']:+.4f}"
                    if args.bootstrap:
                        extra += (f"[{r[f'{m}_ci_low']:+.4f},{r[f'{m}_ci_high']:+.4f}]"
                                  f"p={r[f'{m}_p_boot']:.3g}")
                fdr = f" fdr={r['p_fdr']:.3g}" if args.fdr else ""
                orv = "inf" if r["odds_ratio"] == float("inf") else f"{r['odds_ratio']:.2f}"
                print(f"  {r['a']:<46} vs {r['b']:<46} "
                      f"dacc={r['acc_delta']:+.4f} {r['b01']:>4}/{r['b10']:<4} "
                      f"OR={orv:>5} g={r['cohens_g']:.3f}({r['effect'][:4]}) "
                      f"k={r['kappa']:.2f} p={r['p_mcnemar']:.3g} "
                      f"holm={r['p_holm']:.3g}{fdr} {r['sig']}{extra}")

    total = len(records) + len(per_run_records)

    # Append-only log, on by default: results survive the terminal scrollback and can be
    # re-read, diffed against a later run, or turned into a thesis table without recomputing.
    if not args.no_log and total:
        with open(args.log_file, "a") as f:
            for rec in records + per_run_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nLogged {len(records)} comparisons"
              f"{f' + {len(per_run_records)} per-run rows' if per_run_records else ''}"
              f" to {os.path.relpath(args.log_file, REPO_ROOT)}")

    # Optional frozen snapshot of just this invocation.
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"provenance": meta, "comparisons": records,
                       "per_run": per_run_records}, f, indent=2)
        print(f"Wrote snapshot ({total} rows) to {args.out}")


if __name__ == "__main__":
    main()
