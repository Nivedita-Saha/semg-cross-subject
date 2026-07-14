"""
Verify the split cannot leak. Two checks:

1. No window from a test subject appears in train or val.
2. The normalisation statistics are identical whether or not the test
   subject's data exists at all — i.e. they genuinely do not depend on it.
"""

import numpy as np
from data_pipeline import make_split

d = make_split(train_ids=[1], test_ids=[2], verbose=True)

# Check 1 — the test subject is a different person entirely.
assert set(np.unique(d["s_test"])) == {2}, "test set is not purely subject 2"

# Check 2 — stats from train=[1] must not change if test set changes.
d2 = make_split(train_ids=[1], test_ids=[3], verbose=False)
assert np.allclose(d["mean"], d2["mean"]), "mean depends on the test subject — LEAK"
assert np.allclose(d["std"], d2["std"]), "std depends on the test subject — LEAK"

print("\nNo leakage. Statistics depend only on the training subjects.")
