"""
things_eeg_order.py
PerceptionLab — "Do not hype. Do not lie. Just show."
 
The one test that is about the THEORY, not just about decodability.
 
Field build result (in silico):
   a time-collapsed SUM keeps the SET but loses ORDER (permutation-invariant);
   keeping the TEMPORAL structure keeps ORDER.
Empirical analog on THINGS-EEG, with the RSVP confound controlled:
 
PART 1  Is the stimulus code temporal or static?
        Decode image identity (validation set) from
          (a) time-resolved response  (channels x timepoints)  -- 'phase/time'
          (b) time-AVERAGED response  (channels only)           -- 'spatial snapshot'
        Framework predicts (a) >> (b): identity lives in the waveform, not the mean.
 
PART 2  Can we recover ORDER, and does it live in TIME?
        For adjacent validation images (i before j), recover order by sliding each
        image's leave-out template across the window and comparing peak latency.
          time-resolved  : expect order accuracy >> 0.5
          time-SCRAMBLED : permute the window's time axis, redo -> expect ~0.5
        If order survives time-resolved but collapses when scrambled, order lives
        in temporal structure. HONEST READING: the simplest mechanism is response
        LATENCY (each item peaks at its onset) -- i.e. position == time. That is
        "phase is temporal" in the loosest sense; it is NOT proof of a nontrivial
        phase code, and we will not claim it is.
 
Usage:
  python things_eeg_order.py \
    --eeg E:/.../sub-01_task-rsvp_eeg.vhdr \
    --events E:/.../sub-01_task-rsvp_events.tsv
Run with no args for the synthetic self-test.
"""
import argparse, sys, os
import numpy as np
 
# ---------- fast leave-one-out identity decode (from things_eeg_decode) ----
def loo_decode(X, y, whiten=True):
    classes = np.unique(y); n_cls = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.fromiter((idx[v] for v in y), int, len(y)); n = len(X)
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    if whiten and Z.shape[1] > 1:
        cov = np.cov(Z.T) + 1e-2*np.eye(Z.shape[1])
        vals, vecs = np.linalg.eigh(cov)
        Z = Z @ (vecs @ np.diag(1/np.sqrt(np.maximum(vals, 1e-6))) @ vecs.T)
    sums = np.zeros((n_cls, Z.shape[1])); counts = np.zeros(n_cls)
    np.add.at(sums, yi, Z); np.add.at(counts, yi, 1)
    T = sums / counts[:, None]
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    S = Zn @ Tn.T
    top1 = pair = pairn = 0
    for i in range(n):
        c = yi[i]
        if counts[c] > 1:
            t = (sums[c] - Z[i]) / (counts[c] - 1)
            S[i, c] = Zn[i] @ (t / (np.linalg.norm(t) + 1e-9))
        si = S[i]; top1 += np.sum(si > si[c]) == 0
        pair += np.sum(si[c] > np.delete(si, c)); pairn += n_cls - 1
    return dict(top1=top1/n, pairwise=pair/pairn, chance_top1=1/n_cls, n_classes=n_cls)
 
# ---------- order-by-latency core (pure numpy; self-tested) ----------------
def slide_peak_lag(window, tmpl):
    """Latency (column) where the channel-summed correlation of tmpl with the
    window is maximal. window: (C, Lw), tmpl: (C, Lt)."""
    C, Lw = window.shape; Lt = tmpl.shape[1]
    if Lw < Lt: return 0
    # normalize template per use
    t = tmpl - tmpl.mean()
    best = -1e18; lag = 0
    for L in range(0, Lw - Lt + 1):
        seg = window[:, L:L+Lt]
        c = float(np.sum(seg * t))
        if c > best: best = c; lag = L
    return lag
 
def predict_order(window, tmpl_i, tmpl_j):
    """True if i is predicted to appear before j (its template peaks earlier)."""
    return slide_peak_lag(window, tmpl_i) < slide_peak_lag(window, tmpl_j)
 
def order_accuracy(windows, tis, tjs, scramble=False, seed=0):
    rng = np.random.default_rng(seed); hits = 0
    for w, ti, tj in zip(windows, tis, tjs):
        ww = w
        if scramble:
            perm = rng.permutation(w.shape[1]); ww = w[:, perm]
        hits += predict_order(ww, ti, tj)     # truth is always i-before-j
    return hits / max(1, len(windows))
 
# ======================================================================
# SELF-TEST
# ======================================================================
def self_test():
    print("SELF-TEST (synthetic; no files)")
    rng = np.random.default_rng(0)
    C, Lt, n_img = 8, 60, 20
    sigs = rng.standard_normal((n_img, C, Lt))            # per-image waveform
    # PART 1 synthetic: identity is in the TIME course; the time-MEAN is ~flat
    sigs -= sigs.mean(axis=2, keepdims=True)              # zero time-mean per image
    Xt, Xm, y = [], [], []
    for k in range(n_img):
        for _ in range(10):
            ep = sigs[k] + rng.standard_normal((C, Lt))*1.5
            Xt.append(ep.ravel()); Xm.append(ep.mean(1)); y.append(k)
    Xt, Xm, y = np.array(Xt), np.array(Xm), np.array(y)
    rt = loo_decode(Xt, y); rm = loo_decode(Xm, y)
    print(f"  PART1 identity pairwise: time-resolved={rt['pairwise']:.3f}  "
          f"time-averaged={rm['pairwise']:.3f}  (expect resolved >> averaged)")
 
    # PART 2 synthetic: windows with i placed early, j placed late
    windows, tis, tjs = [], [], []
    Lw = 200
    for _ in range(300):
        i, j = rng.choice(n_img, 2, replace=False)
        w = rng.standard_normal((C, Lw))*1.0
        w[:, 30:30+Lt] += sigs[i]                         # i early
        w[:, 120:120+Lt] += sigs[j]                       # j late
        windows.append(w); tis.append(sigs[i]); tjs.append(sigs[j])
    acc = order_accuracy(windows, tis, tjs, scramble=False)
    accs = order_accuracy(windows, tis, tjs, scramble=True)
    print(f"  PART2 order accuracy: time-resolved={acc:.3f}  scrambled={accs:.3f}  "
          f"(expect resolved>>0.5, scrambled~0.5)")
    ok = rt['pairwise'] > rm['pairwise']+0.1 and acc > 0.8 and abs(accs-0.5) < 0.1
    print(f"  -> order core {'VALID' if ok else 'CHECK'}")
 
# ======================================================================
# REAL DATA
# ======================================================================
def run_real(a):
    import mne, pandas as pd
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(a.eeg, preload=False)
    fs = raw.info["sfreq"]
    picks = [c for c in raw.ch_names if c.upper().startswith(("O", "PO", "IZ"))]
    if not picks: picks = raw.ch_names[:8]
    raw.pick_channels(picks); C = len(picks)
    ev = pd.read_csv(a.events, sep="\t")
    # pick identity column: --label wins; else skip jitter/structural cols and
    # take the tight-repeat categorical partition (-> 'object' on THINGS main task)
    skip = {"onset","duration","sample","trial_type","sample_index","eventnumber",
            "stimnumber","response","rt","correct","istarget","time_stimon",
            "time_stimoff","stimdur","presentationnumber","sequencenumber",
            "blocksequencenumber","withinsequencenumber"}
    idcol = a.label if getattr(a, "label", None) else None
    if idcol is None:
        best_n = 10**9
        for c in ev.columns:
            if c.lower() in skip: continue
            try: vc = ev[c].value_counts(dropna=True)
            except Exception: continue
            hi = vc[vc >= a.min_reps]; n_hi = int(len(hi))
            if n_hi < 20: continue
            tight = float(hi.value_counts().iloc[0]/n_hi)
            if tight >= 0.5 and n_hi < best_n: idcol, best_n = c, n_hi
    if idcol is None or idcol not in ev.columns:
        print(f"[error] no usable identity column; columns={list(ev.columns)}"); sys.exit(1)
    ev = ev.dropna(subset=["onset", idcol]).reset_index(drop=True)
    counts = ev[idcol].value_counts()
    valset = set(counts[counts >= a.min_reps].index.tolist())
    print(f"  picks={picks}  identity column={idcol!r}  classes={len(valset)}")
 
    # onset units
    dur_s = raw.n_times / fs; o = ev["onset"].astype(float).values
    mx = np.nanmax(o)
    if "sample" in ev.columns and ev["sample"].notna().any():
        onset_samp = ev["sample"].astype(float).values
    elif mx <= dur_s*1.5:  onset_samp = o*fs
    elif mx <= raw.n_times*1.5: onset_samp = o
    else: onset_samp = o*fs/1000.0
    onset_samp = onset_samp.astype(int)
    labels = ev[idcol].values
 
    dec = max(1, int(round(fs/250.0)))         # decimate to ~250 Hz
    Lt = int(0.30*fs)//dec                      # 0-300 ms template
    pre = int(0.05*fs); post = int(0.40*fs)
 
    # ---- gather validation epochs (channels x time, decimated) ----
    val_rows = [k for k in range(len(ev)) if labels[k] in valset]
    epochs = {}                                 # row -> (C, Lt)
    for k in val_rows:
        s = onset_samp[k]; e = s + int(0.30*fs)
        if s < 0 or e >= raw.n_times: continue
        d, _ = raw[:, s:e]; epochs[k] = d[:, ::dec][:, :Lt]
    # PART 1 identity decode: time-resolved vs time-averaged
    rowsP1 = [k for k in val_rows if k in epochs]
    idmap = {}; Xt = []; Xm = []; y = []
    for k in rowsP1:
        ep = epochs[k]; Xt.append(ep.ravel()); Xm.append(ep.mean(1))
        y.append(idmap.setdefault(labels[k], len(idmap)))
    Xt, Xm, y = np.array(Xt), np.array(Xm), np.array(y)
    if Xt.shape[1] > 600:
        step = Xt.shape[1]//600 + 1; Xt = Xt[:, ::step]
    rt = loo_decode(Xt, y); rm = loo_decode(Xm, y)
    print("\nPART 1 — is the code temporal or static?")
    print(f"  identity pairwise: time-resolved={rt['pairwise']:.3f}  "
          f"time-averaged={rm['pairwise']:.3f}  chance=0.500")
    print(f"  -> {'TEMPORAL code (resolved > averaged)' if rt['pairwise']>rm['pairwise']+0.02 else 'no temporal advantage'}")
 
    # ---- PART 2: order of adjacent validation images ----
    # per-image template sums (decimated epochs), for leave-one-out
    sums = {}; cnts = {}
    for k in rowsP1:
        c = labels[k]; sums[c] = sums.get(c, 0) + epochs[k]; cnts[c] = cnts.get(c, 0)+1
    # adjacent pairs within validation runs (consecutive events both validation)
    windows, tis, tjs = [], [], []
    for n in range(len(ev)-1):
        a0, b0 = n, n+1
        if labels[a0] in valset and labels[b0] in valset:
            sa, sb = onset_samp[a0], onset_samp[b0]
            if sb <= sa: continue
            if (sb - sa) > 0.30*fs: continue          # genuine adjacent RSVP step only
            w0 = sa - pre; w1 = sb + post
            if w0 < 0 or w1 >= raw.n_times: continue
            if a0 not in epochs or b0 not in epochs: continue
            d, _ = raw[:, w0:w1]; W = d[:, ::dec]
            ci, cj = labels[a0], labels[b0]
            if cnts.get(ci,0) < 2 or cnts.get(cj,0) < 2: continue
            ti = (sums[ci] - epochs[a0])/(cnts[ci]-1)       # leave-out templates
            tj = (sums[cj] - epochs[b0])/(cnts[cj]-1)
            windows.append(W); tis.append(ti); tjs.append(tj)
            if len(windows) >= a.max_pairs: break
    if len(windows) < 20:
        print(f"\nPART 2 — only {len(windows)} adjacent validation pairs found; "
              f"order test underpowered (validation images may not run consecutively).")
        acc = accs = float('nan')
    else:
        acc = order_accuracy(windows, tis, tjs, scramble=False)
        accs = order_accuracy(windows, tis, tjs, scramble=True)
        print(f"\nPART 2 — order of {len(windows)} adjacent validation pairs")
        print(f"  order accuracy: time-resolved={acc:.3f}  time-scrambled={accs:.3f}  chance=0.500")
        verdict = ("order lives in TEMPORAL structure (resolved>>chance, scramble~chance)"
                   if acc > 0.55 and abs(accs-0.5) < 0.05 else
                   "order not cleanly separable from confounds here")
        print(f"  -> {verdict}")
 
    print("\nledger:")
    print("  - PART1 tests the central claim: is identity in the waveform (time) or")
    print("    the static mean (space)? resolved>>averaged == 'phase is temporal'.")
    print("  - PART2 order, IF above chance and IF it dies under time-scramble, lives")
    print("    in temporal structure -- most simply response LATENCY (position==time).")
    print("    That is NOT proof of a nontrivial phase code, and we don't claim it.")
 
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
        ax[0].bar(["time-resolved","time-averaged","chance"],
                  [rt['pairwise'], rm['pairwise'], 0.5],
                  color=["#d95f0e","#2c7fb8","#ccc"]); ax[0].axhline(0.5,color="#666",lw=.8)
        ax[0].set_ylim(0.4,1.0); ax[0].set_title("PART 1: identity code\n(pairwise)")
        if not np.isnan(acc):
            ax[1].bar(["time-resolved","time-scrambled","chance"],[acc,accs,0.5],
                      color=["#d95f0e","#2c7fb8","#ccc"]); ax[1].axhline(0.5,color="#666",lw=.8)
            ax[1].set_ylim(0.4,1.0); ax[1].set_title("PART 2: order\n(accuracy)")
        fig.suptitle("THINGS-EEG: is the code temporal? does order live in time?")
        fig.tight_layout(rect=[0,0,1,0.93])
        out = os.path.join(os.path.dirname(a.eeg) or ".", "things_eeg_order_result.png")
        fig.savefig(out, dpi=130); print(f"\n  saved figure -> {out}")
    except Exception as e:
        print(f"  [figure skipped: {e}]")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg"); ap.add_argument("--events")
    ap.add_argument("--min-reps", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--label", type=str, default=None)
    a = ap.parse_args()
    self_test() if not a.eeg else run_real(a)