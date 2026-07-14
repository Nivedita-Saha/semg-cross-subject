"""
Steps 1.4 + 5.3 — Extract demographics, then ask the closing question:

Are the subjects the model fails on the ones who are anatomically UNLIKE
everyone else?

The first project refuted rotation and scaling as explanations, leaving
anatomy as the last one standing. But "last one standing" is not evidence.
This is the first test that could produce POSITIVE evidence for it.

Prediction: distance from the training subjects (in body-size space)
correlates NEGATIVELY with LOSO accuracy.
"""

import scipy.io
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

rows = []
for s in range(1, 11):
    d = scipy.io.loadmat(f"data/raw/s{s}/S{s}_E2_A1.mat")
    rows.append({
        "subject": s,
        "age": int(d["age"].flatten()[0]),
        "height": int(d["height"].flatten()[0]),
        "weight": int(d["weight"].flatten()[0]),
        "circumference": int(d["circumference"].flatten()[0]),
        "gender": str(d["gender"].flatten()[0]),
        "laterality": str(d["laterality"].flatten()[0]),
    })

df = pd.DataFrame(rows)

loso = {r["held_out"]: r["mean"] for r in json.load(open("results/loso.json"))}
df["loso_acc"] = df["subject"].map(loso)

df.to_csv("data/demographics.csv", index=False)
print(df.to_string(index=False))
print("\nSaved: data/demographics.csv")

# --- Distance from the OTHER nine, in standardised body-size space ---
# LOSO trains on the other nine, so "unlike the training set" means
# "far from the mean of the other nine". Compute it that way, honestly.
feats = ["height", "weight", "circumference"]
Z = (df[feats] - df[feats].mean()) / df[feats].std()

dists = []
for i in range(len(df)):
    others = Z.drop(index=i).mean()          # the nine the model trained on
    dists.append(np.linalg.norm(Z.iloc[i] - others))
df["distance"] = dists

print("\n" + "=" * 66)
print("STEP 5.3 — does anatomical distance predict transfer failure?")
print("=" * 66)
out = df[["subject", "circumference", "height", "weight", "gender",
          "distance", "loso_acc"]].sort_values("loso_acc")
print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\nCorrelations with LOSO accuracy (n=10):")
for name, x in [("distance from other 9", df["distance"]),
                ("circumference", df["circumference"]),
                ("height", df["height"]),
                ("weight", df["weight"])]:
    r, p = pearsonr(x, df["loso_acc"])
    rs, ps = spearmanr(x, df["loso_acc"])
    print(f"  {name:<24} Pearson r={r:+.2f} (p={p:.3f})   "
          f"Spearman rho={rs:+.2f} (p={ps:.3f})")

print("\n  Prediction: distance should correlate NEGATIVELY with accuracy.")
print("  n=10. This is underpowered. Treat as suggestive, never conclusive.")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(df["distance"], df["loso_acc"], s=90, color="#c1121f", zorder=3)
for _, r in df.iterrows():
    ax.annotate(f"s{r['subject']}", (r["distance"], r["loso_acc"]),
                xytext=(6, 4), textcoords="offset points", fontsize=9)
m, b = np.polyfit(df["distance"], df["loso_acc"], 1)
xs = np.linspace(df["distance"].min(), df["distance"].max(), 50)
ax.plot(xs, m * xs + b, "--", color="grey", lw=1.4)
ax.axhline(100/17, color="grey", ls=":", lw=1)
ax.text(df["distance"].max()*0.75, 100/17 + 0.7, "chance", fontsize=8, color="grey")
r, p = pearsonr(df["distance"], df["loso_acc"])
ax.set_xlabel("Body-size distance from the nine training subjects (std. units)")
ax.set_ylabel("LOSO accuracy (%)")
ax.set_title(f"Anatomical distance vs transfer accuracy\nPearson r = {r:+.2f}, p = {p:.3f}, n = 10")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("figures/anatomy_vs_accuracy.png", dpi=170)
print("\nSaved: figures/anatomy_vs_accuracy.png")
