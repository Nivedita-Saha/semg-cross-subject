"""
Phase 2 — The multi-subject pipeline.

Step 2.1: load_subjects(ids) -> X, y, rep_ids, subject_ids

Everything downstream calls this. It cleans, windows, drops rest, and
relabels 1-17 -> 0-16 — the same pipeline as semg-edge-ai, but carrying
a subject_id array alongside the repetition_id array.

Carrying those two arrays IS the point. They are what make an honest
split possible. Without them you cannot split by person or by repetition,
and every accuracy number you produce afterwards is inflated for free.
"""

import numpy as np
import scipy.io

from preprocess import preprocess
from windowing import make_windows

N_CLASSES = 17


def load_subject(sid):
    """Clean and window ONE subject's exercise-2 file."""
    path = f"data/raw/s{sid}/S{sid}_E2_A1.mat"
    d = scipy.io.loadmat(path)

    emg_raw = d["emg"]
    labels = d["restimulus"].flatten()
    reps = d["rerepetition"].flatten()

    emg_clean = preprocess(emg_raw)
    X, y, r = make_windows(emg_clean, labels, reps)

    s = np.full(len(y), sid, dtype=np.int8)
    return X, y, r, s


def load_subjects(ids, drop_rest=True, verbose=True):
    """
    Load a list of subjects and stack them.

    Returns
    -------
    X : (n_windows, 40, 16) float32
    y : (n_windows,) int8   gesture label, 0-16 after relabelling
    r : (n_windows,) int8   repetition id, 1-6
    s : (n_windows,) int8   SUBJECT id — who this window came from
    """
    Xs, ys, rs, ss = [], [], [], []

    for sid in ids:
        if verbose:
            print(f"\nSubject {sid}")
        X, y, r, s = load_subject(sid)
        if verbose:
            print(f"  kept {len(y):,} windows")
        Xs.append(X)
        ys.append(y)
        rs.append(r)
        ss.append(s)

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    r = np.concatenate(rs)
    s = np.concatenate(ss)

    if drop_rest:
        keep = y > 0                      # label 0 is rest, not a gesture
        X, y, r, s = X[keep], y[keep], r[keep], s[keep]
        y = (y - 1).astype(np.int8)       # 1..17 -> 0..16, what Keras expects

    return X, y, r, s


if __name__ == "__main__":

    X, y, r, s = load_subjects([1, 2])

    print("\n" + "=" * 55)
    print("SHAPES")
    print("=" * 55)
    print(f"X : {X.shape}   {X.dtype}")
    print(f"y : {y.shape}   labels {y.min()}-{y.max()}")
    print(f"r : {r.shape}   reps   {sorted(np.unique(r))}")
    print(f"s : {s.shape}   subjs  {sorted(np.unique(s))}")

    print("\nWindows per subject:")
    for sid in np.unique(s):
        print(f"  s{sid}: {(s == sid).sum():,}")

    print("\nWindows per gesture (should be roughly even, no zeros):")
    counts = np.bincount(y, minlength=N_CLASSES)
    print("  " + "  ".join(f"{c}" for c in counts))

    assert y.min() == 0 and y.max() == 16, "labels must be 0-16"
    assert (counts > 0).all(), "some gesture has no windows"
    print("\nChecks passed.")


# ============================================================
# Step 2.2 — The split. Normalisation statistics come from the
# TRAINING SUBJECTS ONLY. This is not a style preference; it is
# the difference between a result and a fiction.
# ============================================================

def make_split(train_ids, test_ids, val_reps=(2,), within_test_reps=(5,), verbose=True):
    """
    Build a cross-subject split.

    Mirrors semg-edge-ai exactly:
        train reps = 1, 3, 4, 6
        val   rep  = 2   (early stopping watches this)
        within-subject test rep = 5  (held out from BOTH -- this is the
                                      repetition that produced 74.13%)

    test_ids are a different PERSON entirely -- all their repetitions.

    Returns train/val/within-test/cross-test, plus the mean/std used.
    """
    Xtr, ytr, rtr, _ = load_subjects(train_ids, verbose=verbose)
    Xte, yte, _, ste = load_subjects(test_ids, verbose=verbose)

    val_mask = np.isin(rtr, val_reps)
    win_mask = np.isin(rtr, within_test_reps)
    trn_mask = ~(val_mask | win_mask)

    X_train, y_train = Xtr[trn_mask], ytr[trn_mask]
    X_val, y_val = Xtr[val_mask], ytr[val_mask]
    X_win, y_win = Xtr[win_mask], ytr[win_mask]

    # Normalisation: fit on TRAIN ONLY, apply to everything else.
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_win = (X_win - mean) / std
    X_test = (Xte - mean) / std        # test subject NEVER touches mean/std

    if verbose:
        print("\n" + "=" * 55)
        print(f"SPLIT  train={list(train_ids)}  test={list(test_ids)}")
        print("=" * 55)
        print(f"  train      : {X_train.shape}  (reps 1,3,4,6)")
        print(f"  val        : {X_val.shape}  (rep {list(val_reps)}, early stopping)")
        print(f"  within-test: {X_win.shape}  (rep {list(within_test_reps)}, unseen reps, SAME people)")
        print(f"  cross-test : {X_test.shape}  (subjects {list(test_ids)}, unseen PERSON)")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_within": X_win, "y_within": y_win,
        "X_test": X_test, "y_test": yte, "s_test": ste,
        "mean": mean, "std": std,
    }