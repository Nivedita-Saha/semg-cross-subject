"""
Phase 3 — The scaling experiment. The core of the project.

Subjects 9 and 10 are the PERMANENT held-out test set. They are never
trained on and never tuned on. Locked.

Train on N = 1, 2, 4, 6, 8 subjects drawn from subjects 1-8.
For each N: 3 subject draws x 3 runs. (At N=8 only one draw exists.)

Record BOTH:
  cross-subject accuracy  -> on s9+s10 (the question)
  within-subject accuracy -> rep 5 of the TRAINING subjects (the control)

The control matters. If within-subject accuracy also falls as N grows,
the model is failing to FIT many anatomies (capacity), not failing to
GENERALISE to a new one. Different diagnosis, different fix.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json, time, os

from data_pipeline import make_split_cached
from model import build_model

POOL = [1, 2, 3, 4, 5, 6, 7, 8]      # may train on these
HELD_OUT = [9, 10]                   # LOCKED. Never train. Never tune.

N_VALUES = [1, 2, 4, 6, 8]
N_DRAWS = 3
N_RUNS = 3
DRAW_RNG = np.random.default_rng(1234)   # draws reproducible; TF noise is not

os.makedirs("results", exist_ok=True)
OUT = "results/scaling.json"

records = []
t0 = time.time()

for N in N_VALUES:
    # At N=8 there is exactly one possible draw. Do not fake a spread.
    n_draws = 1 if N == len(POOL) else N_DRAWS

    # Draws must be DISTINCT from one another. Sampling independently can
    # return the same subject set twice (it did: N=1 drew [8] three times),
    # which silently collapses the spread we are trying to measure.
    draws = []
    guard = 0
    while len(draws) < n_draws and guard < 500:
        cand = sorted(DRAW_RNG.choice(POOL, size=N, replace=False).tolist())
        if cand not in draws:
            draws.append(cand)
        guard += 1

    for draw_i, train_ids in enumerate(draws):
        d = make_split_cached(train_ids=train_ids, test_ids=HELD_OUT)

        for run in range(N_RUNS):
            keras.backend.clear_session()
            model = build_model(input_shape=d["X_train"].shape[1:])
            model.fit(
                d["X_train"], d["y_train"],
                validation_data=(d["X_val"], d["y_val"]),
                epochs=150, batch_size=32,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True)],
                verbose=0,
            )

            _, cross = model.evaluate(d["X_test"], d["y_test"], verbose=0)
            _, within = model.evaluate(d["X_within"], d["y_within"], verbose=0)

            records.append({
                "N": N, "draw": draw_i, "run": run,
                "train_ids": train_ids,
                "cross": cross * 100,
                "within": within * 100,
                "n_train_windows": int(d["X_train"].shape[0]),
            })

            print(f"N={N}  draw {draw_i} {str(train_ids):<26} run {run}  "
                  f"cross {cross*100:5.2f}%   within {within*100:5.2f}%")

    # summary for this N
    rows = [r for r in records if r["N"] == N]
    c = np.array([r["cross"] for r in rows])
    w = np.array([r["within"] for r in rows])
    print(f"  --> N={N}: cross {c.mean():5.2f} +/- {c.std():.2f} | "
          f"within {w.mean():5.2f} +/- {w.std():.2f}   ({len(rows)} models)\n")

with open(OUT, "w") as f:
    json.dump(records, f, indent=2)

print("=" * 62)
print(f"{len(records)} models trained in {(time.time()-t0)/60:.1f} min")
print(f"Saved: {OUT}")
print("=" * 62)
for N in N_VALUES:
    rows = [r for r in records if r["N"] == N]
    c = np.array([r["cross"] for r in rows])
    w = np.array([r["within"] for r in rows])
    print(f"  N={N:<2}  cross {c.mean():5.2f} +/- {c.std():4.2f}   "
          f"within {w.mean():5.2f} +/- {w.std():4.2f}")
print("\nChance: 5.88%")
