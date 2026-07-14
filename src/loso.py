"""
Phase 5 — Leave-one-subject-out.

Train on 9 subjects, test on the 10th. Repeat ten times.

This is the standard protocol in the sEMG literature. Using it makes these
numbers directly comparable to published DB5 results — and it replaces a
two-person test set with a ten-person one.

The per-subject SPREAD is as important as the mean. If accuracy depends
heavily on WHO is held out, the variance across people is the finding.
"""

import numpy as np
from tensorflow import keras
import json, os, time

from data_pipeline import make_split_cached
from model import build_model

SUBJECTS = list(range(1, 11))
N_RUNS = 3

records = []
t0 = time.time()

for held_out in SUBJECTS:
    train_ids = [s for s in SUBJECTS if s != held_out]
    d = make_split_cached(train_ids=train_ids, test_ids=[held_out])

    accs = []
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
        _, acc = model.evaluate(d["X_test"], d["y_test"], verbose=0)
        accs.append(acc * 100)

    a = np.array(accs)
    records.append({"held_out": held_out, "accs": accs,
                    "mean": a.mean(), "std": a.std()})
    print(f"  hold out s{held_out:<3} {a.mean():5.2f} +/- {a.std():4.2f} %"
          f"   runs: {['%.1f' % x for x in accs]}")

os.makedirs("results", exist_ok=True)
with open("results/loso.json", "w") as f:
    json.dump(records, f, indent=2)

means = np.array([r["mean"] for r in records])

print("\n" + "=" * 60)
print("LEAVE-ONE-SUBJECT-OUT  (train on 9, test on the 10th)")
print("=" * 60)
print(f"  Mean across subjects : {means.mean():.2f} %")
print(f"  Std  across subjects : {means.std():.2f} pp")
print(f"  Range                : {means.min():.2f} % (s{records[int(means.argmin())]['held_out']})"
      f"  to  {means.max():.2f} % (s{records[int(means.argmax())]['held_out']})")
print(f"  Spread               : {means.max()-means.min():.2f} pp")
print(f"  Chance               :  5.88 %")
print(f"\n  {len(records)*N_RUNS} models in {(time.time()-t0)/60:.1f} min")
print("\n  If the spread across PEOPLE dwarfs the spread across RUNS,")
print("  the variance between individuals is itself the result.")
