"""
Step 1.3 — Integrity check.

Load all ten subjects' E2 files and confirm they are structurally identical:
same 17 gestures, same 6 repetitions, same 16 channels.

If one subject is malformed, we want to know NOW — not after a four-hour
training run whose result we cannot explain.
"""

import scipy.io
import numpy as np

SUBJECTS = list(range(1, 11))
FS = 200

print(f"{'Subj':<6}{'Samples':>10}{'Chans':>7}{'Gestures':>10}{'Reps':>6}   {'Minutes':>8}")
print("-" * 55)

summary = []

for s in SUBJECTS:
    path = f"data/raw/s{s}/S{s}_E2_A1.mat"
    d = scipy.io.loadmat(path)

    emg = d["emg"]
    labels = d["restimulus"].flatten()
    reps = d["rerepetition"].flatten()

    # Exclude 0 — that is rest, not a gesture / not a repetition.
    gestures = sorted(set(np.unique(labels)) - {0})
    repetitions = sorted(set(np.unique(reps)) - {0})

    n_samples, n_chans = emg.shape
    minutes = n_samples / FS / 60

    print(f"s{s:<5}{n_samples:>10,}{n_chans:>7}{len(gestures):>10}{len(repetitions):>6}   {minutes:>7.1f}")

    summary.append({
        "subject": s,
        "gestures": gestures,
        "reps": repetitions,
        "chans": n_chans,
    })

# --- Now assert they all agree. A quiet pass here is the whole point. ---
print()
ref = summary[0]
problems = []

for r in summary:
    if r["gestures"] != ref["gestures"]:
        problems.append(f"s{r['subject']}: gesture labels differ from s1 -> {r['gestures']}")
    if r["reps"] != ref["reps"]:
        problems.append(f"s{r['subject']}: repetition ids differ from s1 -> {r['reps']}")
    if r["chans"] != ref["chans"]:
        problems.append(f"s{r['subject']}: {r['chans']} channels, expected {ref['chans']}")

if problems:
    print("PROBLEMS FOUND — do not proceed:")
    for p in problems:
        print("  " + p)
else:
    print("All ten subjects agree.")
    print(f"  Gestures    : {ref['gestures']}  ({len(ref['gestures'])} classes)")
    print(f"  Repetitions : {ref['reps']}")
    print(f"  Channels    : {ref['chans']}")
    print(f"  Chance level: {100/len(ref['gestures']):.2f}%")
