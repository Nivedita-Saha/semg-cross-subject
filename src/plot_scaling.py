"""
Phase 4 — The scaling curve. The headline figure.

Plots:
  - cross-subject accuracy vs number of training subjects, with error bars
  - within-subject accuracy on the same axes (the control)
  - a saturating fit, and what it extrapolates to

The shape IS the finding.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CHANCE = 100 / 17
TARGET = 60.0          # the accuracy a deployable system would want

recs = json.load(open("results/scaling.json"))
Ns = sorted({r["N"] for r in recs})

cross_m, cross_s, within_m, within_s = [], [], [], []
for N in Ns:
    rows = [r for r in recs if r["N"] == N]
    c = np.array([r["cross"] for r in rows])
    w = np.array([r["within"] for r in rows])
    cross_m.append(c.mean()); cross_s.append(c.std())
    within_m.append(w.mean()); within_s.append(w.std())

Ns = np.array(Ns, float)
cross_m, cross_s = np.array(cross_m), np.array(cross_s)
within_m, within_s = np.array(within_m), np.array(within_s)


# --- Saturating fit:  acc = A - B * exp(-k * N)   (A = asymptote) ---
def saturating(N, A, B, k):
    return A - B * np.exp(-k * N)

fit_ok = True
try:
    popt, _ = curve_fit(saturating, Ns, cross_m, p0=[25, 10, 0.5],
                        bounds=([0, 0, 0.01], [100, 100, 5]), maxfev=20000)
    A, B, k = popt
except Exception as e:
    fit_ok = False
    print("Fit failed:", e)

fig, ax = plt.subplots(figsize=(9, 6))

# within-subject: the control
ax.errorbar(Ns, within_m, yerr=within_s, marker="s", ms=7, lw=2,
            capsize=4, color="#2a6f97",
            label="Within-subject (unseen repetitions, same people)")

# cross-subject: the question
ax.errorbar(Ns, cross_m, yerr=cross_s, marker="o", ms=8, lw=2.5,
            capsize=4, color="#c1121f",
            label="Cross-subject (unseen PEOPLE, s9 + s10)")

if fit_ok:
    xs = np.linspace(1, 30, 300)
    ax.plot(xs, saturating(xs, A, B, k), "--", lw=1.5, color="#c1121f",
            alpha=0.6, label=f"Saturating fit (asymptote = {A:.1f}%)")
    ax.axhline(A, color="#c1121f", lw=0.8, ls=":", alpha=0.5)

ax.axhline(CHANCE, color="grey", lw=1, ls="--")
ax.text(8.3, CHANCE + 0.8, f"chance ({CHANCE:.2f}%)", fontsize=9, color="grey")

ax.axhline(TARGET, color="darkgreen", lw=1, ls="--", alpha=0.6)
ax.text(8.3, TARGET + 1.2, f"target ({TARGET:.0f}%)", fontsize=9, color="darkgreen")

ax.set_xlabel("Number of training subjects", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Cross-subject accuracy does not scale with the number of training subjects\n"
             "Ninapro DB5 · 17 gestures · 1D-CNN · 3 subject draws x 3 runs per point",
             fontsize=12, pad=14)
ax.set_xticks(Ns)
ax.set_xlim(0.5, 8.8)
ax.set_ylim(0, 80)
ax.grid(alpha=0.25)
ax.legend(loc="center right", fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig("figures/scaling_curve.png", dpi=170)
print("Saved: figures/scaling_curve.png")


# ---------------- The numbers, stated plainly ----------------
print("\n" + "=" * 64)
print("THE SCALING CURVE")
print("=" * 64)
print(f"  {'N':<4}{'cross-subject':>20}{'within-subject':>20}")
for i, N in enumerate(Ns):
    print(f"  {int(N):<4}{cross_m[i]:>12.2f} +/-{cross_s[i]:5.2f}"
          f"{within_m[i]:>12.2f} +/-{within_s[i]:5.2f}")

print(f"\n  Cross-subject, N=1 -> N=8 : {cross_m[-1]-cross_m[0]:+.2f} pp")
print(f"  Within-subject, N=1 -> N=8: {within_m[-1]-within_m[0]:+.2f} pp")
print(f"  Gap at N=1 : {within_m[0]-cross_m[0]:.1f} pp")
print(f"  Gap at N=8 : {within_m[-1]-cross_m[-1]:.1f} pp")
print("  (the gap narrows mainly because the TOP line falls, not because")
print("   the bottom line rises — do not report this as 'the gap closes')")

if fit_ok:
    print(f"\n  Saturating fit: acc = {A:.1f} - {B:.1f} * exp(-{k:.2f} N)")
    print(f"  Asymptote (infinite subjects): {A:.1f}%")
    if A >= TARGET:
        n_needed = -np.log((A - TARGET) / B) / k
        print(f"  Subjects needed to reach {TARGET:.0f}%: {n_needed:.0f}")
    else:
        print(f"\n  Subjects needed to reach {TARGET:.0f}%: NO FINITE ANSWER.")
        print(f"  The fitted asymptote ({A:.1f}%) lies BELOW the target ({TARGET:.0f}%).")
        print("  On this evidence, adding subjects does not reach a deployable")
        print("  accuracy at any N. Data scale is not the answer to this problem.")
print("=" * 64)
