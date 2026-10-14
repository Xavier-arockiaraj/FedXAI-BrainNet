#!/usr/bin/env python3
"""
FedXAI-BrainNet - Section 6 simulation study (synthetic proxy dataset)
=======================================================================

Reproduces every experiment in Section 6 of the manuscript
"FedXAI-BrainNet: A Federated, Uncertainty-Aware, and Explainable Lightweight
Framework for Trustworthy Fetal Brain Plane Classification".

    6.1  Synthetic proxy dataset (1,560 images, 3 classes, 3 non-IID sites)   -> Fig. 3
    6.2  Surrogate encoder: 264-D features + MLP (64-32), scikit-learn Adam
    6.3  Centralized vs. federated (FedAvg, Eq. 1; optional DP, Eq. 2)         -> Figs. 4, 5
    6.4  Uncertainty: 15-member bootstrap ensemble (MC-Dropout proxy, Eqs. 3-4)
         ECE, variance/error correlation, deferral curve                        -> Fig. 6a
    6.5  Occlusion deletion-AUC fidelity (proxy for Eqs. 5-6) and
         speckle-noise robustness sweep                                         -> Figs. 6b, 7
    6.6  INT8 post-training quantization (Eq. 7), model size and latency
    Table 4 summary                                                             -> table4_simulation_summary.csv

Usage
-----
    python fedxai_brainnet_simulation.py                # default settings, seed 42
    python fedxai_brainnet_simulation.py --dp-sigma 0.5 # add Gaussian DP noise to client updates
    python fedxai_brainnet_simulation.py --out results_run2 --seed 7

Requirements: Python >= 3.9, numpy, scipy, scikit-learn, matplotlib.
All outputs (figures, CSV, JSON) are written to the --out folder.
"""
import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

CLASSES = ["Trans-ventricular", "Trans-thalamic", "Trans-cerebellar"]
IMG = 64
CONTRAST = 0.25  # amplitude scale of class-defining internal structures (lower = harder)
JITTER = 2.5     # positional jitter scale of internal structures

# Site configuration (Section 6.1). Per-site class counts are skewed (non-IID)
# while every class totals 520 images overall.
SITES = {
    "A": {"noise": 0.28, "counts": [260, 220, 180]},   # 660 images
    "B": {"noise": 0.42, "counts": [140, 160, 180]},   # 480 images
    "C": {"noise": 0.55, "counts": [120, 140, 160]},   # 420 images
}


# --------------------------------------------------------------------------
# 6.1 Synthetic proxy image generator
# --------------------------------------------------------------------------
def _ellipse(xx, yy, cx, cy, ax, ay, ang=0.0):
    c, s = np.cos(ang), np.sin(ang)
    xr = (xx - cx) * c + (yy - cy) * s
    yr = -(xx - cx) * s + (yy - cy) * c
    return (xr / ax) ** 2 + (yr / ay) ** 2


def _blob(xx, yy, cx, cy, ax, ay, ang=0.0, amp=1.0):
    return amp * np.exp(-_ellipse(xx, yy, cx, cy, ax, ay, ang))


def make_image(cls, noise, rng):
    """One 64x64 synthetic fetal-head image of class `cls` with speckle severity `noise`."""
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(float)
    cx, cy = 32 + rng.normal(0, 1.5), 32 + rng.normal(0, 1.5)
    ax, ay = 25 + rng.normal(0, 1.2), 21 + rng.normal(0, 1.2)
    ang = rng.normal(0, 0.15)

    r = _ellipse(xx, yy, cx, cy, ax, ay, ang)
    img = np.where(r <= 1.0, 0.32, 0.05)                       # brain parenchyma vs background
    img += 0.65 * np.exp(-((np.sqrt(r) - 1.0) / 0.06) ** 2)    # bright skull ring

    j = lambda s=1.0: rng.normal(0, s * JITTER)                # geometric jitter
    if cls == 0:   # trans-ventricular: paired, elongated lateral blobs
        for side in (-1, 1):
            img += _blob(xx, yy, cx + side * (8 + j()), cy - 3 + j(), 3.0, 7.5, ang + side * 0.25, 0.55 * CONTRAST)
    elif cls == 1:  # trans-thalamic: symmetric round thalamic blobs near the midline
        for side in (-1, 1):
            img += _blob(xx, yy, cx + side * (4.5 + j(0.5)), cy + 2 + j(), 3.6, 4.2, ang, 0.55 * CONTRAST)
    else:           # trans-cerebellar: posterior dumbbell (two hemispheres + vermis bridge)
        py = cy + 11 + j()
        for side in (-1, 1):
            img += _blob(xx, yy, cx + side * (6.5 + j(0.6)), py, 3.8, 3.4, ang, 0.55 * CONTRAST)
        img += _blob(xx, yy, cx, py, 5.5, 1.3, ang, 0.35 * CONTRAST)

    k = 1.0 / noise ** 2                                       # gamma speckle: mean 1, var noise^2
    img = img * rng.gamma(k, 1.0 / k, size=img.shape)
    img = img + rng.normal(0, 0.04, size=img.shape)            # additive sensor noise
    return np.clip(img, 0, 1).astype(np.float32)


def make_site(counts, noise, rng):
    X, y = [], []
    for c, n in enumerate(counts):
        for _ in range(n):
            X.append(make_image(c, noise, rng))
            y.append(c)
    return np.stack(X), np.array(y)


# --------------------------------------------------------------------------
# 6.2 Surrogate encoder features (264-D) and MLP
# --------------------------------------------------------------------------
def features(imgs):
    """16x16 block-mean intensity map (256) + 8-bin gradient-magnitude histogram (8) = 264."""
    n = len(imgs)
    down = imgs.reshape(n, 16, 4, 16, 4).mean(axis=(2, 4)).reshape(n, 256)
    gy, gx = np.gradient(imgs, axis=(1, 2))
    gm = np.sqrt(gx ** 2 + gy ** 2).reshape(n, -1)
    edges = np.linspace(0, 0.5, 9)
    hist = np.stack([np.histogram(np.clip(g, 0, 0.4999), bins=edges)[0] for g in gm]).astype(np.float32)
    hist /= hist.sum(axis=1, keepdims=True)
    return np.hstack([down, hist]).astype(np.float64)


def new_mlp(seed, max_iter=300):
    return MLPClassifier(hidden_layer_sizes=(64, 32), solver="adam", learning_rate_init=1e-3,
                         alpha=1e-4, batch_size=32, max_iter=max_iter, random_state=seed)


def get_w(m):
    return [w.copy() for w in m.coefs_] + [b.copy() for b in m.intercepts_]


def set_w(m, ws):
    L = len(m.coefs_)
    m.coefs_ = [w.copy() for w in ws[:L]]
    m.intercepts_ = [b.copy() for b in ws[L:]]


def forward(ws, X):
    """Plain NumPy forward pass (ReLU hidden layers, softmax output)."""
    L = len(ws) // 2
    h = X
    for i in range(L):
        h = h @ ws[i] + ws[L + i]
        if i < L - 1:
            h = np.maximum(h, 0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "precision": pr, "recall": rc, "f1": f1}


# --------------------------------------------------------------------------
# 6.3 Federated averaging (Eq. 1) with optional Gaussian DP (Eq. 2)
# --------------------------------------------------------------------------
def fedavg(site_data, rounds, local_epochs, seed, dp_sigma, dp_clip, test_sets, rng):
    gm = new_mlp(seed, max_iter=1)
    # data-free initialisation of weight shapes (no client data leaves a site)
    gm.partial_fit(np.zeros((3, 264)), np.array([0, 1, 2]), classes=np.array([0, 1, 2]))
    gw = get_w(gm)
    n_total = sum(len(d[1]) for d in site_data.values())
    history = {s: [] for s in site_data}

    for rnd in range(rounds):
        agg = [np.zeros_like(w) for w in gw]
        for k, (s, (Xk, yk)) in enumerate(site_data.items()):
            local = copy.deepcopy(gm)
            local.set_params(max_iter=local_epochs, warm_start=True, random_state=seed + 100 * rnd + k)
            set_w(local, gw)
            local.fit(Xk, yk)
            lw = get_w(local)
            if dp_sigma > 0:   # Eq. 2: clip the update to norm C, add N(0, sigma^2 C^2 I)
                delta = [a - b for a, b in zip(lw, gw)]
                norm = np.sqrt(sum((d ** 2).sum() for d in delta))
                scale = min(1.0, dp_clip / (norm + 1e-12))
                lw = [b + d * scale + rng.normal(0, dp_sigma * dp_clip, size=d.shape)
                      for b, d in zip(gw, delta)]
            wk = len(yk) / n_total                     # Eq. 1 sample weighting n_k / n
            agg = [a + wk * w for a, w in zip(agg, lw)]
        gw = agg
        set_w(gm, gw)
        for s, (Xt, yt) in test_sets.items():
            history[s].append(accuracy_score(yt, gm.predict(Xt)))
    return gm, history


# --------------------------------------------------------------------------
# 6.4 Uncertainty helpers
# --------------------------------------------------------------------------
def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


# --------------------------------------------------------------------------
# 6.5 Occlusion saliency and deletion-AUC
# --------------------------------------------------------------------------
def occlusion_saliency(model_w, img, cls, patch=8):
    base = forward(model_w, features(img[None]))[0, cls]
    n = IMG // patch
    sal = np.zeros((n, n))
    fill = img.mean()
    for i in range(n):
        for j in range(n):
            im2 = img.copy()
            im2[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch] = fill
            sal[i, j] = base - forward(model_w, features(im2[None]))[0, cls]
    return sal


def deletion_auc(model_w, img, cls, sal, patch=8):
    order = np.argsort(-sal.ravel())
    n = IMG // patch
    im2 = img.copy()
    fill = img.mean()
    probs = [forward(model_w, features(im2[None]))[0, cls]]
    for idx in order:
        i, j = divmod(idx, n)
        im2[i * patch:(i + 1) * patch, j * patch:(j + 1) * patch] = fill
        probs.append(forward(model_w, features(im2[None]))[0, cls])
    probs = np.array(probs)
    return float(np.trapezoid(probs, dx=1.0 / (len(probs) - 1))), probs


# --------------------------------------------------------------------------
# 6.6 INT8 quantization (Eq. 7)
# --------------------------------------------------------------------------
def quantize(w):
    """Per-tensor asymmetric uint8: q = round(x/s) + z, s = (xmax - xmin)/255."""
    xmin, xmax = float(w.min()), float(w.max())
    s = (xmax - xmin) / 255.0 if xmax > xmin else 1.0
    z = int(round(-xmin / s))
    q = np.clip(np.round(w / s) + z, 0, 255).astype(np.uint8)
    return q, s, z


def dequantize(q, s, z):
    return (q.astype(np.float64) - z) * s


def latency_ms(ws, x, reps=200):
    forward(ws, x)  # warm-up
    t = time.perf_counter()
    for _ in range(reps):
        forward(ws, x)
    return (time.perf_counter() - t) / reps * 1e3


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--rounds", type=int, default=6, help="federated communication rounds")
    ap.add_argument("--local-epochs", type=int, default=40, help="local epochs per round")
    ap.add_argument("--dp-sigma", type=float, default=0.0, help="DP noise multiplier sigma (0 = off)")
    ap.add_argument("--dp-clip", type=float, default=5.0, help="DP clipping norm C")
    ap.add_argument("--ensemble", type=int, default=15, help="bootstrap ensemble size (T)")
    ap.add_argument("--contrast", type=float, default=0.25, help="structure contrast (lower = harder task)")
    ap.add_argument("--jitter", type=float, default=2.5, help="structure position jitter (higher = harder task)")
    a = ap.parse_args()
    global CONTRAST, JITTER
    CONTRAST, JITTER = a.contrast, a.jitter
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    R = {"settings": vars(a)}

    # ---- 6.1 data -------------------------------------------------------
    raw, tr, te = {}, {}, {}
    for s, cfg in SITES.items():
        X, y = make_site(cfg["counts"], cfg["noise"], rng)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)
        raw[s] = (Xtr, Xte, ytr, yte)
        tr[s] = (features(Xtr), ytr)
        te[s] = (features(Xte), yte)
    Xtr_all = np.vstack([tr[s][0] for s in SITES]); ytr_all = np.concatenate([tr[s][1] for s in SITES])
    Xte_all = np.vstack([te[s][0] for s in SITES]); yte_all = np.concatenate([te[s][1] for s in SITES])
    img_te_all = np.vstack([raw[s][1] for s in SITES])
    R["dataset"] = {"total": int(sum(sum(c["counts"]) for c in SITES.values())),
                    "test": int(len(yte_all)),
                    "per_site": {s: int(sum(c["counts"])) for s, c in SITES.items()}}
    print(f"[6.1] {R['dataset']['total']} images, pooled test set = {len(yte_all)}")

    # Fig. 3: exemplars
    fig, axs = plt.subplots(3, 6, figsize=(9, 4.8))
    for c in range(3):
        for k in range(6):
            s = "ABC"[k // 2]
            Xs, ys = raw[s][0], raw[s][2]
            axs[c, k].imshow(Xs[np.where(ys == c)[0][k % 2]], cmap="gray", vmin=0, vmax=1)
            axs[c, k].axis("off")
        axs[c, 0].set_title(CLASSES[c], fontsize=8, loc="left")
    plt.suptitle("Synthetic proxy exemplars (columns: Site A, A, B, B, C, C)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig3_synthetic_exemplars.png", dpi=200); plt.close()

    # ---- 6.3 centralized ------------------------------------------------
    cen = new_mlp(a.seed).fit(Xtr_all, ytr_all)
    p_cen = cen.predict(Xte_all)
    R["centralized"] = metrics(yte_all, p_cen)
    print("[6.3] centralized:", {k: round(v, 3) for k, v in R["centralized"].items()})

    # ---- 6.3 federated --------------------------------------------------
    fed, hist = fedavg(tr, a.rounds, a.local_epochs, a.seed, a.dp_sigma, a.dp_clip, te, rng)
    p_fed = fed.predict(Xte_all)
    R["federated"] = metrics(yte_all, p_fed)
    R["federated"]["per_site_accuracy"] = {s: accuracy_score(te[s][1], fed.predict(te[s][0])) for s in SITES}
    R["federated"]["round_history"] = hist
    print("[6.3] federated:", {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in R["federated"].items() if k != "round_history"})

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    short = ["Trans-vent.", "Trans-thal.", "Trans-cereb."]
    for ax, p, t in zip(axs, [p_cen, p_fed], ["(a) Centralized single-pool training",
                                              f"(b) Federated aggregation ({len(SITES)} simulated sites)"]):
        cm = confusion_matrix(yte_all, p)
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(3), short, fontsize=8); ax.set_yticks(range(3), short, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(t, fontsize=9)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig4_confusion_matrices.png", dpi=200); plt.close()

    plt.figure(figsize=(5.5, 3.8))
    for s in SITES:
        plt.plot(range(1, a.rounds + 1), hist[s], marker="o", label=f"Site_{s}")
    plt.xlabel("Federated communication round"); plt.ylabel("Local validation accuracy")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{a.out}/fig5_federated_rounds.png", dpi=200); plt.close()

    # ---- 6.4 uncertainty (bootstrap ensemble as MC-Dropout proxy) -------
    probs = []
    for t in range(a.ensemble):
        idx = rng.integers(0, len(ytr_all), len(ytr_all))
        m = new_mlp(a.seed + 1000 + t).fit(Xtr_all[idx], ytr_all[idx])
        probs.append(m.predict_proba(Xte_all))
    probs = np.stack(probs)                                 # T x N x C
    p_mean = probs.mean(0)                                  # Eq. 3
    var = ((probs - p_mean) ** 2).mean(0).sum(1)            # Eq. 4, summed over classes
    pred = p_mean.argmax(1)
    correct = (pred == yte_all)
    err = (~correct).astype(float)
    fr = np.arange(0, 0.51, 0.05)
    order = np.argsort(-var)
    defer_acc = []
    for f in fr:
        keep = np.ones(len(var), bool); keep[order[:int(round(f * len(var)))]] = False
        defer_acc.append(correct[keep].mean())
    R["uncertainty"] = {
        "ensemble_accuracy": correct.mean(),
        "mean_var_correct": var[correct].mean(),
        "mean_var_incorrect": var[~correct].mean() if (~correct).any() else float("nan"),
        "corr_var_error": float(np.corrcoef(var, err)[0, 1]) if err.std() > 0 else float("nan"),
        "ece_10bin": ece(p_mean.max(1), correct.astype(float)),
        "accuracy_after_deferring_20pct": defer_acc[list(np.round(fr, 2)).index(0.2)],
        "deferral_curve": {f"{f:.2f}": v for f, v in zip(fr, defer_acc)},
    }
    print("[6.4] uncertainty:", {k: round(v, 3) for k, v in R["uncertainty"].items() if k != "deferral_curve"})

    # ---- 6.5 explainability fidelity and robustness ---------------------
    cw = get_w(cen)
    sel = rng.choice(len(yte_all), 25, replace=False)
    aucs, sals = [], []
    for i in sel:
        c = int(cen.predict(Xte_all[i:i + 1])[0])
        sal = occlusion_saliency(cw, img_te_all[i], c)
        auc, _ = deletion_auc(cw, img_te_all[i], c, sal)
        aucs.append(auc); sals.append((i, c, sal))
    noise_levels = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    rob = []
    for nz in noise_levels:
        Xn, yn = make_site([100, 100, 100], nz, rng)
        rob.append(accuracy_score(yn, cen.predict(features(Xn))))
    R["explainability"] = {"mean_deletion_auc": float(np.mean(aucs)), "n_images": len(sel)}
    R["robustness"] = {f"{n:.2f}": v for n, v in zip(noise_levels, rob)}
    print(f"[6.5] mean deletion-AUC = {np.mean(aucs):.3f}; robustness:",
          {k: round(v, 3) for k, v in R["robustness"].items()})

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].plot(fr * 100, np.array(defer_acc) * 100, marker="o")
    axs[0].set_xlabel("Cases deferred to sonographer (%)"); axs[0].set_ylabel("Accuracy on retained cases (%)")
    axs[0].set_title("(a) Uncertainty-based referral", fontsize=9); axs[0].grid(alpha=0.3)
    axs[1].plot(noise_levels, np.array(rob) * 100, marker="s", color="tab:red")
    axs[1].set_xlabel("Speckle-noise severity"); axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_title("(b) Robustness to acquisition noise", fontsize=9); axs[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig6_uncertainty_robustness.png", dpi=200); plt.close()

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, (i, c, sal) in zip(axs, sals[:4]):
        p = forward(cw, Xte_all[i:i + 1])[0, c]
        ax.imshow(img_te_all[i], cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.kron(np.clip(sal, 0, None), np.ones((8, 8))), cmap="jet", alpha=0.45)
        ax.set_title(f"{CLASSES[c]}\np = {p:.2f}", fontsize=8); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig7_saliency_overlays.png", dpi=200); plt.close()

    # ---- 6.6 INT8 quantization and latency ------------------------------
    qws = [dequantize(*quantize(w)) for w in cw]
    n_params = sum(w.size for w in cw)
    size_fp = n_params * cw[0].itemsize / 1024
    size_q = n_params * 1 / 1024
    acc_q = accuracy_score(yte_all, forward(qws, Xte_all).argmax(1))
    x1 = Xte_all[:1]
    R["quantization"] = {"n_params": int(n_params), "size_fp_kb": size_fp, "size_int8_kb": size_q,
                         "compression": size_fp / size_q,
                         "acc_fp": R["centralized"]["accuracy"], "acc_int8": acc_q,
                         "latency_fp_ms": latency_ms(cw, x1), "latency_int8_ms": latency_ms(qws, x1)}
    print("[6.6] quantization:", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in R["quantization"].items()})

    # ---- Table 4 ---------------------------------------------------------
    c, f, u, q = R["centralized"], R["federated"], R["uncertainty"], R["quantization"]
    rows = [
        ("Accuracy", f"{c['accuracy']:.3f}", f"{f['accuracy']:.3f}", f"{u['ensemble_accuracy']:.3f}", f"{q['acc_int8']:.3f}"),
        ("Precision (macro)", f"{c['precision']:.3f}", f"{f['precision']:.3f}", "-", "-"),
        ("Recall (macro)", f"{c['recall']:.3f}", f"{f['recall']:.3f}", "-", "-"),
        ("F1-score (macro)", f"{c['f1']:.3f}", f"{f['f1']:.3f}", "-", "-"),
        ("Model size", f"{q['size_fp_kb']:.1f} KB", f"{q['size_fp_kb']:.1f} KB", "-",
         f"{q['size_int8_kb']:.1f} KB ({q['compression']:.1f}x smaller)"),
        ("Mean single-image latency (CPU)", f"{q['latency_fp_ms']:.3f} ms", f"{q['latency_fp_ms']:.3f} ms", "-",
         f"{q['latency_int8_ms']:.3f} ms"),
    ]
    with open(f"{a.out}/table4_simulation_summary.csv", "w") as fh:
        fh.write("Metric,Centralized baseline,Federated (3 sites),MC-Dropout ensemble,INT8 quantized\n")
        for r in rows:
            fh.write(",".join(f'"{v}"' for v in r) + "\n")

    def tojson(o):
        if isinstance(o, dict): return {k: tojson(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [tojson(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    with open(f"{a.out}/results.json", "w") as fh:
        json.dump(tojson(R), fh, indent=2)
    print(f"\nAll outputs written to ./{a.out}/")


if __name__ == "__main__":
    main()
