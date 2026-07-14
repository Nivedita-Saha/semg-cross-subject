# semg-cross-subject

**The generalisation study.** How much does cross-subject accuracy improve per additional training subject — and where does it stop improving?

Follow-on to [semg-edge-ai](https://github.com/nivsaha/semg-edge-ai) (the compression study). Ninapro DB5, 10 subjects, 17 gestures, 1D-CNN.

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

## Status

Work in progress.
