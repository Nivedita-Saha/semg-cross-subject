"""
Milestone 1, properly — three seeds.

A single run cannot distinguish 'the pipeline is sound' from 'the pipeline
is broken and I got lucky'. The spread across seeds can.

Expected (semg-edge-ai, single seed): within 74.13%, cross 27.62%.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_pipeline import make_split
from model import build_model

SEEDS = [0, 1, 2]

d = make_split(train_ids=[1], test_ids=[2], verbose=True)

within_scores, cross_scores = [], []

for seed in SEEDS:
    np.random.seed(seed)
    tf.random.set_seed(seed)
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

    _, w = model.evaluate(d["X_within"], d["y_within"], verbose=0)
    _, c = model.evaluate(d["X_test"], d["y_test"], verbose=0)
    within_scores.append(w * 100)
    cross_scores.append(c * 100)
    print(f"  seed {seed}: within {w*100:5.2f}%   cross {c*100:5.2f}%")

w = np.array(within_scores)
c = np.array(cross_scores)

print("\n" + "=" * 58)
print("MILESTONE 1 — three seeds, mean +/- std")
print("=" * 58)
print(f"  Within-subject (s1, rep 5) : {w.mean():5.2f} +/- {w.std():.2f} %   expected ~74.13%")
print(f"  Cross-subject  (s2, unseen): {c.mean():5.2f} +/- {c.std():.2f} %   expected ~27.62%")
print(f"  Chance                     :  5.88 %")
print(f"\n  Gap : {w.mean() - c.mean():.2f} pp")
print("\nIf the expected values sit within roughly 2 std of these means,")
print("the pipeline reproduces the old result. Proceed.")
