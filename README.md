# semg-cross-subject

**The generalisation study.** How much does cross-subject accuracy improve per additional training subject — and where does it stop improving?

Follow-on to [semg-edge-ai](https://github.com/Nivedita-Saha/semg-edge-ai) (the compression study). Ninapro DB5, 10 subjects, 17 gestures, 1D-CNN.

## Where this project starts

The first project compressed a 1D-CNN 4.4x with negligible accuracy loss — then found the model did not transfer to a new person. Three explanations for that failure were proposed and tested. All three were refuted:

| Hypothesis | Test | Result |
|---|---|---|
| Armband rotation (make the model ignore it) | Electrode-shift augmentation, 3 seeds | **Refuted** — -14.1 pp within-subject, -11.0 pp cross-subject |
| Amplitude / scaling differences | Three normalisation strategies | **Refuted** — 7% of the gap; rest-based calibration actively harmful (-13.8 pp) |
| Armband rotation (correct it) | Oracle over all 8 possible shifts | **Refuted** — the best rotation is no rotation; the ceiling is zero |

The conclusion those force: the cross-subject gap is **not geometric and not a scaling artefact**. It is anatomical. Different forearms produce genuinely different spatial signatures for the same intended gesture — not rotated versions, not rescaled versions, different ones.

So multi-subject training cannot teach a universal gesture pattern; there isn't one. At best it teaches a *distribution over anatomies*. That is interpolation, not generalisation — and it should saturate.

**This project measures where.**

## Baselines to beat

| | Accuracy |
|---|---|
| Within-subject (train s1, test s1) | 74.13% |
| Cross-subject (train s1, test s2) | 27.62% |
| Chance (17 classes) | 5.88% |

## Headline result

![Scaling curve](figures/scaling_curve.png)

**More subjects do not fix cross-subject transfer.** Cross-subject accuracy climbs from 17.6% (N=1) to 24.6% (N=4), then stops. N=6 and N=8 are indistinguishable from N=4. A saturating fit puts the asymptote at **25.0%** — below any deployable threshold — so there is no number of training subjects at which the data-only approach reaches 60%.

Within-subject accuracy meanwhile *falls* (71.9% -> 60.3%) as subjects are added. The gap narrows because the top line descends, not because the bottom line rises.

| N subjects | Cross-subject | Within-subject |
|---|---|---|
| 1 | 17.59 +/- 2.11 | 71.92 +/- 3.00 |
| 2 | 18.12 +/- 2.64 | 67.15 +/- 1.89 |
| 4 | 24.62 +/- 1.54 | 66.03 +/- 1.69 |
| 6 | 24.42 +/- 1.69 | 61.74 +/- 1.31 |
| 8 | 23.94 +/- 1.50 | 60.30 +/- 1.38 |

3 subject draws x 3 runs per point. Subjects 9 and 10 held out permanently.

## The obvious objection, ruled out

*"Your CNN was simply too small to hold eight forearms."* Tested: widen the network at fixed N=8.

| Width | Params | Within-subject | Cross-subject |
|---|---|---|---|
| 1x | 18,545 | 61.01 +/- 0.79 | 23.48 +/- 1.57 |
| 2x | 65,745 | 65.69 +/- 0.32 | 23.94 +/- 0.55 |
| 4x | 246,161 | 66.71 +/- 1.72 | 22.24 +/- 0.74 |

13x the parameters recovers within-subject accuracy (+5.7 pp) and leaves cross-subject accuracy unchanged. **The plateau is not a capacity artefact.** The extra capacity is spent memorising training anatomies; none of it transfers.

## Status

Phases 0-4 complete. LOSO (all ten subjects) and few-shot calibration in progress.
