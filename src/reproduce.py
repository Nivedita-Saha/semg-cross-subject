"""
Step 2.3 — Milestone 1.

Train on s1, test on s2. Must reproduce semg-edge-ai:
    within-subject  ~74.13%
    cross-subject   ~27.62%

If it does not, the new pipeline has a bug. Do not proceed.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_pipeline import make_split
from model import build_model

SEED = 42

# model.py sets seed 42 at import time. Re-seed explicitly here so the
# seed is something WE control, not a side effect of an import.
np.random.seed(SEED)
tf.random.set_seed(SEED)

d = make_split(train_ids=[1], test_ids=[2])

model = build_model(input_shape=d["X_train"].shape[1:])

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

print("\nTraining...\n")
model.fit(
    d["X_train"], d["y_train"],
    validation_data=(d["X_val"], d["y_val"]),
    epochs=150,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
)

# --- Within-subject: rep 5 — held out from training AND early stopping ---
_, within = model.evaluate(d["X_within"], d["y_within"], verbose=0)

# --- Cross-subject: a different person entirely ---
_, cross = model.evaluate(d["X_test"], d["y_test"], verbose=0)

print("\n" + "=" * 55)
print("MILESTONE 1 — does the new pipeline reproduce the old result?")
print("=" * 55)
print(f"  Within-subject (s1 rep5)  : {within*100:5.2f}%   expected ~74.13%")
print(f"  Cross-subject  (s2 test)  : {cross*100:5.2f}%   expected ~27.62%")
print(f"  Chance                    :  5.88%")
print()
print(f"  Gap                       : {(within-cross)*100:5.2f} pp")
print()
print("Within a few points of expected = pipeline is sound, proceed.")
print("Wildly different = there is a bug. Stop and find it.")
