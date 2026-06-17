"""
things_eeg_decode.py
PerceptionLab — "Do not hype. Do not lie. Just show."

The honest empirical readout test, replacing the circular hologram loop.

THE TRAP we are avoiding:
  empirical_hologram_loop.json feeds the (occluded) IMAGE into the field and
  reads a blurred IMAGE back, then correlates with the clean image. That is an
  image autoencoder; the EEG only gates it. It will always "match" and proves
  nothing about the brain.

THE REAL QUESTION:
  Can we retrieve the image a subject saw FROM THE EEG ALONE -- image never fed
  into the reconstruction? This exercises exactly the two framework operations
  that survived the field test:
     RECORD  = superposition: average the repeats of each validation image into
               one neural template e_k  (the 'plate')
     READ    = projection: correlate a held-out single trial x against every e_k
               and take argmax  ->  the retrieved image
  Score against chance (1/n_images) and against a label-shuffled null. If
  retrieval beats chance, the EEG plate genuinely supports projection readout of
  stimulus identity. It does NOT prove "the brain is a hologram" -- it proves the
  occipital response carries decodable identity that a projection reader can pull
  out, which is the strongest honest form of the claim this data can support.

  Baseline matters: we also report a plain whitened-correlation (LDA-ish) reader.
  If the framework's diffusive smoothing adds nothing over plain projection, we
  say so.

Targets the 200 validation images (repeated 12x) -- the subset THINGS-EEG built
for exactly this. Occipital/parieto-occipital channels, 50-300 ms window.

Usage:
  pip install mne pandas numpy scipy matplotlib
  python things_eeg_decode.py \
      --eeg   E:/DocsHouse/811/ds003825/sub-01/eeg/sub-01_task-rsvp_eeg.vhdr \
      --events E:/DocsHouse/811/ds003825/sub-01/eeg/sub-01_task-rsvp_events.tsv \
      --images-root E:/DocsHouse/811/THINGS-database/osfstorage/01_image-level/object_images \
      --images-csv  E:/DocsHouse/811/THINGS-database/osfstorage/01_image-level/image-paths.csv
"""
import argparse, os, sys
import numpy as np

# ======================================================================
# PURE-NUMPY DECODE CORE  (self-tested below; no mne/files needed)
# ======================================================================
def build_templates(X, y, train_mask):
    """Average training trials of each class into a template (the 'plate').
    X: (n_trials, n_features), y: (n_trials,) int labels, train_mask: bool."""
    classes = np.unique(y)
    T = np.zeros((len(classes), X.shape[1]))
    for i, c in enumerate(classes):
        m = train_mask & (y == c)
        if m.any():
            T[i] = X[m].mean(0)
    return classes, T

def zscore_fit(X):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return mu, sd

def retrieve(X_test, y_test, classes, T, mu, sd, whiten=None):
    """Projection readout: correlate each test trial with every template; argmax
    is the retrieved class. Returns rank of the true class per trial (0=top-1)."""
    Xt = (X_test - mu) / sd
    Tt = (T - mu) / sd
    if whiten is not None:                      # optional whitening (LDA-ish)
        Xt = Xt @ whiten; Tt = Tt @ whiten
    # cosine similarity
    Xn = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-9)
    Tn = Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Tn.T                                # (n_test, n_classes)
    ranks = []
    for i in range(len(X_test)):
        true_i = np.where(classes == y_test[i])[0][0]
        order = np.argsort(-S[i])
        ranks.append(int(np.where(order == true_i)[0][0]))
    return np.array(ranks), S

def loo_decode(X, y, whiten=False):
    """Vectorized leave-one-repeat-out retrieval.
    Key speedups vs the naive version:
      - z-score and whitening computed ONCE on all trials (removing one trial of
        2400 changes them negligibly);
      - a leave-one-out class template is just (class_sum - held)/(n-1), a cheap
        rank-1 correction, not a full recompute;
      - all trial-vs-template cosine similarities are one matmul.
    Returns top1, top5, mean pairwise accuracy, and chance."""
    classes = np.unique(y); n_cls = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.fromiter((idx[v] for v in y), int, len(y))
    n = len(X)

    mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    if whiten:
        cov = np.cov(Z.T) + 1e-2*np.eye(Z.shape[1])      # once, not per-fold
        vals, vecs = np.linalg.eigh(cov)
        W = vecs @ np.diag(1.0/np.sqrt(np.maximum(vals, 1e-6))) @ vecs.T
        Z = Z @ W

    F = Z.shape[1]
    sums = np.zeros((n_cls, F)); counts = np.zeros(n_cls)
    np.add.at(sums, yi, Z); np.add.at(counts, yi, 1)
    T = sums / counts[:, None]                            # full templates
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Zn @ Tn.T                                         # (n, n_cls) one matmul

    top1 = top5 = 0; pair = 0; pairn = 0
    for i in range(n):
        c = yi[i]
        if counts[c] > 1:                                # LOO correction (own class)
            t = (sums[c] - Z[i]) / (counts[c] - 1)
            S[i, c] = Zn[i] @ (t / (np.linalg.norm(t) + 1e-9))
        si = S[i]; rank = int(np.sum(si > si[c]))        # how many beat the truth
        top1 += rank == 0; top5 += rank < 5
        pair += np.sum(si[c] > np.delete(si, c)); pairn += n_cls - 1
    return dict(top1=top1/n, top5=top5/n, pairwise=pair/pairn,
                chance_top1=1.0/n_cls, chance_pair=0.5, n_classes=n_cls, n_test=n)

def shuffle_null(X, y, n=10, whiten=False):
    rng = np.random.default_rng(0); accs = []
    for _ in range(n):
        accs.append(loo_decode(X, rng.permutation(y), whiten)["top1"])
    return float(np.mean(accs)), float(np.std(accs))

# ======================================================================
# SELF-TEST  (runs when no --eeg given): synthetic EEG with known identity
# ======================================================================
def self_test():
    print("SELF-TEST (synthetic EEG; no data files needed)")
    rng = np.random.default_rng(1)
    n_img, n_rep, n_feat = 20, 12, 200
    signatures = rng.standard_normal((n_img, n_feat))     # each image's true ERP
    X, y = [], []
    for k in range(n_img):
        for _ in range(n_rep):
            X.append(signatures[k] + rng.standard_normal(n_feat)*3.0)  # noisy
            y.append(k)
    X = np.array(X); y = np.array(y)
    res = loo_decode(X, y)
    nullm, nulls = shuffle_null(X, y, n=10)
    print(f"  signal : top1={res['top1']:.3f} top5={res['top5']:.3f} "
          f"pairwise={res['pairwise']:.3f}  (chance top1={res['chance_top1']:.3f})")
    print(f"  shuffled-null top1 = {nullm:.3f} +/- {nulls:.3f}")
    # pure noise -> should be at chance
    Xn = rng.standard_normal(X.shape)
    rn = loo_decode(Xn, y)
    print(f"  pure-noise control: top1={rn['top1']:.3f} pairwise={rn['pairwise']:.3f} "
          f"(expected ~chance)")
    ok = res['top1'] > 5*res['chance_top1'] and abs(rn['pairwise']-0.5) < 0.05
    print(f"  -> decode core {'VALID' if ok else 'CHECK'}")

# ======================================================================
# REAL DATA PATH (mne + files)
# ======================================================================
def choose_identity_column(ev, min_reps, override=None):
    """Pick a stimulus-identity column by its repeat FINGERPRINT.
    --label OVERRIDE always wins. Otherwise: a real identity column has a TIGHT
    repeat distribution (validation images: ~200 labels all at 12; concept: 1854
    all at 12). A timing-jitter column like 'stimdur' is SMEARED and is rejected."""
    if override:
        return override, "FORCED BY --label"
    skip = {"onset", "duration", "sample", "trial_type", "sample_index",
            "eventnumber", "stimnumber", "response", "rt", "correct", "istarget",
            "time_stimon", "time_stimoff", "stimdur", "presentationnumber",
            "sequencenumber", "blocksequencenumber", "withinsequencenumber"}
    best, best_n, report = None, 10**9, {}
    for c in ev.columns:
        if c.lower() in skip:
            continue
        try:
            vc = ev[c].value_counts(dropna=True)
        except Exception:
            continue
        hi = vc[vc >= min_reps]
        n_hi = int(len(hi))
        if n_hi < 20:
            report[c] = n_hi; continue
        tight = float(hi.value_counts().iloc[0] / n_hi)   # frac sharing modal count
        report[c] = f"{n_hi}@tight{tight:.2f}"
        if tight >= 0.5 and n_hi < best_n:
            best, best_n = c, n_hi
    return best, report

def run_real(a):
    import mne, pandas as pd
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(a.eeg, preload=False)
    fs = raw.info["sfreq"]
    picks = [c for c in raw.ch_names if c.upper().startswith(("O", "PO", "IZ"))]
    if not picks: picks = raw.ch_names[:8]
    raw.pick_channels(picks)
    print(f"  fs={fs}  occipital picks={picks}")

    ev = pd.read_csv(a.events, sep="\t")
    if "onset" not in ev.columns:
        print("[error] events has no 'onset' column"); sys.exit(1)
    idcol, report = choose_identity_column(ev, a.min_reps, getattr(a, "label", None))
    if idcol is None:
        print(f"[error] no usable identity column; columns={list(ev.columns)}"); sys.exit(1)
    print(f"  identity column: {idcol!r}  (high-rep-label counts per column: {report})")
    ev = ev.dropna(subset=["onset", idcol])

    # validation set = labels with >= min_reps reps in the chosen column
    counts = ev[idcol].value_counts()
    repeated = counts[counts >= a.min_reps].index.tolist()
    if len(repeated) > a.max_images:
        repeated = repeated[:a.max_images]
    print(f"  using {len(repeated)} validation images (>= {a.min_reps} reps each, "
          f"median reps {int(counts[repeated].median())})")

    # epoch window 50-300 ms
    w0, w1 = int(0.05*fs), int(0.30*fs)
    # --- resolve onset units robustly ---------------------------------
    dur_s = raw.n_times / fs
    sub = ev[ev[idcol].isin(repeated)].copy()
    if "sample" in ev.columns and ev["sample"].notna().any():
        onset_samp = sub["sample"].astype(float).values            # already samples
        unit = "sample-column"
    else:
        o = sub["onset"].astype(float).values
        mx = np.nanmax(o)
        if mx <= dur_s * 1.5:                 # onset in SECONDS (BIDS spec)
            onset_samp = o * fs; unit = "seconds"
        elif mx <= raw.n_times * 1.5:         # onset already in SAMPLES
            onset_samp = o; unit = "samples"
        else:                                  # onset in MILLISECONDS
            onset_samp = o * fs / 1000.0; unit = "milliseconds"
    print(f"  onset units detected: {unit}  (max onset {np.nanmax(onset_samp):.0f} "
          f"vs n_times {raw.n_times})")

    labels = sub[idcol].values
    X, y, idmap = [], [], {}
    for k in range(len(sub)):
        s = int(onset_samp[k]) + w0; e = int(onset_samp[k]) + w1
        if s < 0 or e >= raw.n_times: continue
        d, _ = raw[:, s:e]                          # (chan, time)
        X.append(d.ravel())
        y.append(idmap.setdefault(labels[k], len(idmap)))
    X = np.array(X); y = np.array(y)

    if X.ndim != 2 or len(X) == 0:
        print("\n[error] built 0 usable epochs. diagnostics:")
        print(f"  idcol={idcol!r}  events columns={list(ev.columns)}")
        print(f"  onset head: {ev['onset'].head().tolist()}")
        print(f"  n_times={raw.n_times}  fs={fs}  window=[{w0},{w1}] samples")
        print("  -> if onset units look wrong, tell me the onset head values.")
        sys.exit(1)

    # downsample features if huge
    if X.shape[1] > 600:
        step = X.shape[1]//600 + 1; X = X[:, ::step]
    print(f"  trials={len(X)}  feature-dim={X.shape[1]}  images={len(idmap)}")

    res = loo_decode(X, y); resw = loo_decode(X, y, whiten=True)
    nullm, nulls = shuffle_null(X, y, n=20)
    print("\nRESULT (EEG -> image retrieval, image never fed in):")
    print(f"  plain  projection : top1={res['top1']:.3f} top5={res['top5']:.3f} "
          f"pairwise={res['pairwise']:.3f}")
    print(f"  whitened projection: top1={resw['top1']:.3f} top5={resw['top5']:.3f} "
          f"pairwise={resw['pairwise']:.3f}")
    print(f"  chance top1={res['chance_top1']:.4f}  pairwise=0.500")
    print(f"  shuffled-null top1 = {nullm:.4f} +/- {nulls:.4f}")
    print("\nledger:")
    print("  - image is NEVER fed into the reader; this is pure EEG->identity.")
    print("  - RECORD=superposition(avg repeats), READ=projection(correlate) -- the")
    print("    two framework ops that survived the field test, on real brain data.")
    print("  - whitening is the standard baseline; if it beats plain projection, the")
    print("    diffusive-smoothing story adds nothing here and we say so.")
    print("  - above chance => occipital response carries projection-readable identity.")
    print("    It does NOT prove 'the brain is a hologram'.")

    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
        ax[0].bar(["plain", "whitened", "shuffle null", "chance"],
                  [res['top1'], resw['top1'], nullm, res['chance_top1']],
                  color=["#d95f0e", "#2c7fb8", "#999", "#ccc"])
        ax[0].errorbar([2], [nullm], yerr=[nulls], color="k")
        ax[0].set_title(f"top-1 retrieval of {res['n_classes']} images\n(EEG only)")
        ax[0].set_ylabel("accuracy")
        ax[1].bar(["plain", "whitened", "chance"],
                  [res['pairwise'], resw['pairwise'], 0.5],
                  color=["#d95f0e", "#2c7fb8", "#ccc"])
        ax[1].axhline(0.5, color="#666", lw=0.8); ax[1].set_ylim(0.4, 1.0)
        ax[1].set_title("pairwise decoding\n(true image vs each other)")
        fig.suptitle("THINGS-EEG: image identity retrieved from occipital EEG alone")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(os.path.dirname(a.eeg) or ".", "things_eeg_decode_result.png")
        fig.savefig(out, dpi=130); print(f"\n  saved figure -> {out}")
    except Exception as e:
        print(f"  [figure skipped: {e}]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg"); ap.add_argument("--events")
    ap.add_argument("--images-root", default=""); ap.add_argument("--images-csv", default="")
    ap.add_argument("--min-reps", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=250)
    ap.add_argument("--label", type=str, default=None)
    a = ap.parse_args()
    if not a.eeg:
        self_test()
    else:
        run_real(a)