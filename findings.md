# Findings — The Cross-Subject Scaling Study

**Ninapro DB5 · 17 gestures · 1D-CNN · follow-on to [semg-edge-ai](https://github.com/Nivedita-Saha/semg-edge-ai)**

## The question

The first project found that a compressed 1D-CNN did not transfer from one person to another, and refuted three explanations for why (armband rotation, twice; amplitude scaling). That left one explanation standing: the cross-subject gap is anatomical, not geometric. But "the last explanation standing" is not evidence for it.

This study asks a question with a shape rather than a yes/no: **how much does cross-subject accuracy improve per additional training subject, and where does it stop?**

## Headline finding

**Training on more subjects does not solve cross-subject transfer. It saturates at four subjects, at roughly a quarter of the accuracy a deployable system needs — and the plateau is not caused by the model being too small.**

## The five results

| # | Hypothesis / question | Test | Result |
|---|---|---|---|
| 1 | More training subjects will steadily improve cross-subject transfer | Scaling curve, N = 1,2,4,6,8, up to 3 draws x 3 runs, held-out subjects 9+10 | **Refuted** — rises 17.6% -> 24.6% by N=4, then flat (N=6: 24.4%, N=8: 23.9%). Saturating fit asymptote ≈ 25% (95% CI 20.4-29.7%) |
| 2 | The plateau is an artefact of the model being too small to hold many anatomies | Widen the CNN 13x (18.5k -> 246k params) at fixed N=8 | **Refuted** — within-subject recovers +5.7 pp, cross-subject unchanged (23.5% -> 22.2%). Extra capacity is spent memorising training anatomies |
| 3 | Cross-subject accuracy is roughly the same for everyone | Leave-one-subject-out across all 10 subjects, 3 runs each | **Refuted** — mean 25.35%, but ranges 12.6% (s10) to 35.3% (s2), a 22.7 pp spread. Between-person variance is 7.4x run-to-run variance |
| 4 | The subjects the model fails on are those most unlike the training set in body size | Correlate LOSO accuracy with distance from the 9 training subjects in (height, weight, forearm circumference) space, and with each measure separately | **Refuted** — composite distance r = -0.07, p = 0.84, n=10. Individually: BMI r = -0.05, height r = +0.09, weight r = +0.04, forearm circumference r = +0.41 (p = 0.24). None significant; no visible trend |
| 5 | The scaling-curve plateau is an artefact of the specific s9+s10 test set | Compare the 2-subject held-out result against the 10-subject LOSO mean | **Not supported** — LOSO mean (25.35%) matches the scaling asymptote to within 0.4 pp. s9 and s10 are among the ten, so the figures are not independent, but they sit at opposite ends of the LOSO range (33.2% and 12.6%), so the pair was neither an unusually easy nor an unusually hard test set |

## What the results force

The gap is not a data-quantity problem: eight subjects do no better than four. It is not a model-capacity problem: 13x the parameters does not move it. It is a **person** problem — which individual you transfer *to* matters far more than how many people you trained on (a 22.7 pp swing across held-out subjects, against ~1 pp of run-to-run noise).

But it is not explained by gross body size. Height, weight, BMI, and forearm circumference — the anatomical measurements DB5 provides — show no detectable relationship with transfer accuracy (result 4). BMI is worth naming explicitly: the DB5 dataset paper reports that BMI significantly affected classification accuracy in 3 of 10 feature-classifier combinations (Pizzolato *et al.*, 2017), so it was a live candidate. It does not predict cross-subject transfer here.

So the difficulty is anatomical in a *finer* sense than body size: muscle architecture, subcutaneous fat distribution and the cross-talk it produces (Kuiken, Lowery and Stoykov, 2003), the precise placement of electrodes on an individual forearm, motor-unit recruitment habits (Farina, Cescon and Merletti, 2002). None of these is captured by the four numbers in the dataset. This sharpens the first project's conclusion rather than confirming it: "anatomy" cannot mean simply "body size."

The Ninapro authors reached a compatible conclusion from the raw signal rather than from classification: their MANOVA on sEMG amplitude finds no significant effect of movement repetition but significant differences between subjects, which they attribute to subjects having different anatomical features and muscular characteristics (Atzori *et al.*, 2014; Pizzolato *et al.*, 2017). Between-subject variation is a documented property of these data, not an artefact of this model.

## The scaling curve

![Scaling curve](figures/scaling_curve.png)

| N subjects | Cross-subject | Within-subject |
|---|---|---|
| 1 | 17.59 +/- 2.11 | 71.92 +/- 3.00 |
| 2 | 18.12 +/- 2.64 | 67.15 +/- 1.89 |
| 4 | 24.62 +/- 1.54 | 66.03 +/- 1.69 |
| 6 | 24.42 +/- 1.69 | 61.74 +/- 1.31 |
| 8 | 23.94 +/- 1.50 | 60.30 +/- 1.38 |

Error bars at N=1-6 combine subject-draw and run-to-run variance; at N=8 only one draw exists (the full pool), so its error bar reflects run-to-run variance alone.

Within-subject accuracy *falls* as subjects are added. The gap between the curves narrows (54 -> 36 pp) almost entirely because the upper curve descends, not because the lower one rises. Reporting this as "the gap closes" would be misleading.

## Extrapolation

The saturating fit `acc = A - B·exp(-k·N)` places the asymptote A at approximately 25%, with an approximate 95% confidence interval of 20.4%–29.7% — below any deployable threshold. Asking "how many subjects to reach 60%?" therefore has **no finite answer** on this architecture and dataset: the fitted ceiling lies below the target, and so does the upper bound of its confidence interval. The conclusion does not depend on the point estimate. Data scale is not the route to a deployable cross-subject model here.

## How these numbers sit against the literature

The published baseline for DB5 is 69.04%, obtained with the double-Myo setup using the mDWT feature and an SVM classifier across 41 movements (Pizzolato *et al.*, 2017). That is a **within-subject** figure: the classifier is trained and tested on the same people, on different repetitions. The within-subject column above (60–72% across 17 gestures) sits in the same regime, so this study's within-subject performance is in line with the dataset's own baseline rather than below it.

The comparable quantity for the question asked here — leave-one-subject-out, where the test person was never trained on — is not reported in the DB5 paper, and is far lower wherever it is reported. This study's 25.35% LOSO mean is roughly a third of the within-subject figure on the same data, same pipeline, same model. The distance between within-subject numbers and cross-subject reality is the central unsolved problem in the field, not a shortcoming of this particular model.

## Limitations, stated plainly

- **The extrapolation rests on five points and a three-parameter fit.** "No finite N" is a claim about this architecture on this dataset, not a law. The confidence interval on the asymptote (20.4–29.7%) is wide.
- **The subject pool is small and demographically narrow.** DB5 contains 10 subjects, 8 male and 2 female, aged 22–34, all right-handed. Sex cannot be tested as a factor at n=2. A wider population would likely widen the spread rather than narrow it, but that is an expectation, not a measurement.
- **N=8 is the ceiling of the design.** With 8 subjects in the training pool, the plateau is observed over N=4 to N=8 only. That 8 is the maximum available is a limit of the dataset, not evidence about larger N.
- **The anatomy conclusion is by elimination, not by direct measurement.** Result 4 rules out body size; it does not measure the finer anatomical factors it points to. Confirming those would need per-subject electrode-level or imaging data that DB5 does not contain.
- **n=10 makes result 4 underpowered.** A weak body-size effect could be hidden — forearm circumference reaches r = +0.41 without approaching significance. The claim is "no detectable relationship and no visible trend," not "proven absence."

## What this sets up

The practical answer the first project pointed to — a small amount of labelled data from the new user (few-shot calibration) — is now the obvious next experiment, because the data-only route is closed. If scale cannot cross the person gap, per-user calibration is the only remaining mechanism with a clear rationale.

## References

Atzori, M., Gijsberts, A., Castellini, C., Caputo, B., Mittaz Hager, A.-G., Elsig, S., Giatsidis, G., Bassetto, F. and Müller, H. (2014) 'Electromyography data for non-invasive naturally-controlled robotic hand prostheses', *Scientific Data*, 1, 140053.

Farina, D., Cescon, C. and Merletti, R. (2002) 'Influence of anatomical, physical, and detection-system parameters on surface EMG', *Biological Cybernetics*, 86(6), pp. 445–456.

Kuiken, T.A., Lowery, M.M. and Stoykov, N.S. (2003) 'The effect of subcutaneous fat on myoelectric signal amplitude and cross-talk', *Prosthetics and Orthotics International*, 27(1), pp. 48–54.

Pizzolato, S., Tagliapietra, L., Cognolato, M., Reggiani, M., Müller, H. and Atzori, M. (2017) 'Comparison of six electromyography acquisition setups on hand movement classification tasks', *PLoS ONE*, 12(10), e0186132.
