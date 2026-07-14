"""
Step 2.4 — Clean all ten subjects once, cache to a single .npz.

Cleaning is slow. Phase 3 trains ~39 models. Do this once.
"""

import numpy as np
import os
from data_pipeline import load_subjects

os.makedirs("data/processed", exist_ok=True)
OUT = "data/processed/db5_all10_e2.npz"

X, y, r, s = load_subjects(list(range(1, 11)), verbose=True)

np.savez_compressed(OUT, X=X, y=y, r=r, s=s)

print("\n" + "=" * 55)
print(f"Cached: {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
print("=" * 55)
print(f"  X : {X.shape}")
print(f"  windows per subject:")
for sid in np.unique(s):
    print(f"    s{sid:<3}: {(s == sid).sum():,}")
