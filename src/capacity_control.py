"""
Step 3.5 — Is the plateau anatomy, or is it capacity?

The scaling experiment showed within-subject accuracy FALLING 11.6 pp as
subjects were added. That is the signature of a model too small to fit
many anatomies at once — a capacity problem, not a generalisation one.

If that is what is happening, the ~24% cross-subject plateau is an
artefact of model size and the headline claim collapses.

Test: hold N=8 fixed, widen the network. Then:
  within recovers, cross stays flat  -> capacity ruled out. Anatomy stands.
  within recovers, cross climbs      -> the plateau was OUR fault. Rethink.
  neither recovers                   -> not capacity. Something else.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json, os

from data_pipeline import make_split_cached

POOL = [1, 2, 3, 4, 5, 6, 7, 8]
HELD_OUT = [9, 10]
WIDTHS = [1, 2, 4]        # 1x = the baseline architecture, unchanged
N_RUNS = 3
N_CLASSES = 17


def build_wide(input_shape, width=1):
    """The SAME architecture as model.py, scaled by `width`. Nothing else changes."""
    m = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(32 * width, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64 * width, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64 * width, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


d = make_split_cached(train_ids=POOL, test_ids=HELD_OUT)
print(f"N=8, train windows: {d['X_train'].shape[0]:,}\n")

records = []

for width in WIDTHS:
    for run in range(N_RUNS):
        keras.backend.clear_session()
        model = build_wide(d["X_train"].shape[1:], width=width)
        params = model.count_params()

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

        records.append({"width": width, "run": run, "params": params,
                        "cross": cross * 100, "within": within * 100})
        print(f"  width {width}x ({params:>7,} params)  run {run}  "
              f"cross {cross*100:5.2f}%   within {within*100:5.2f}%")

    rows = [r for r in records if r["width"] == width]
    c = np.array([r["cross"] for r in rows])
    w = np.array([r["within"] for r in rows])
    print(f"  --> {width}x: cross {c.mean():5.2f} +/- {c.std():4.2f} | "
          f"within {w.mean():5.2f} +/- {w.std():4.2f}\n")

os.makedirs("results", exist_ok=True)
with open("results/capacity_control.json", "w") as f:
    json.dump(records, f, indent=2)

print("=" * 62)
print("CAPACITY CONTROL  (N=8 fixed, only model width changes)")
print("=" * 62)
print(f"  {'width':<8}{'params':>10}{'within':>18}{'cross':>18}")
for width in WIDTHS:
    rows = [r for r in records if r["width"] == width]
    c = np.array([r["cross"] for r in rows])
    w = np.array([r["within"] for r in rows])
    print(f"  {str(width)+'x':<8}{rows[0]['params']:>10,}"
          f"{w.mean():>11.2f} +/-{w.std():5.2f}"
          f"{c.mean():>11.2f} +/-{c.std():5.2f}")
print("\n  Reference: N=1 within ~71.9%   |   N=8 cross plateau ~23.9%")
print("\n  If within climbs back toward 70% while cross stays ~24%,")
print("  the plateau is NOT a capacity artefact. The anatomy story holds.")
