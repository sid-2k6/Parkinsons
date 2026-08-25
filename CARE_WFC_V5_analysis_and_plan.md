# What your actual run says, and what I'm doing about it

## What I found in the uploaded notebook's saved outputs

I read the outputs already baked into `CARE_WindFarmC_EventBag_V4 (1) (1).ipynb` — your D3/D5/D6
diagnostics and the full 58-event pooled comparison actually ran. That is real information, not
something I'm inferring:

**D6 (leakage control) passed.** The classical baseline's real ROC-AUC (0.687) clears the
permutation-null p95 (0.644), `p = 0.020`. So whatever signal exists is real, not an artifact.

**D5 (classical baseline): ROC-AUC 0.687 / PR-AUC 0.690.** A plain L2-logistic-regression on
hand-crafted per-channel quantile features, evaluated on the same asset folds as the deep models.

**MAIN comparison (58 pooled out-of-fold events):**

| Model | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|
| ModernTCN (adapted) | 0.621 | 0.612 | 0.082 |
| TimeMixer++ (adapted) | 0.593 | 0.612 | 0.007 |
| iTransformer (adapted) | 0.541 | 0.499 | -0.004 |
| **MS-WTFormer (Proposed)** | 0.539 | 0.560 | 0.136 |
| Plain Transformer | 0.485 | 0.501 | 0.062 |
| Non-stationary Transformer | 0.472 | 0.509 | 0.016 |
| *classical baseline (D5)* | *0.690* | *0.687* | — |

**Every deep model's PR-AUC is at or below the plain classical baseline.** That is the actual
finding. It is not that MS-WTFormer is a bit behind ModernTCN — every neural architecture here,
including the proposed one, currently loses to a linear model on hand-crafted features.

**Training curves confirm why:** Plain Transformer fold 0 seed 1 — training loss collapses from
0.775 to 0.0004 by epoch 12, while validation event ROC-AUC oscillates between 0.29 and 0.37 the
entire time and never tracks the loss. That is textbook overfitting on ~30-40 training events.

**MS-WTFormer's 95% CI is [0.430, 0.723].** The CI is wider than the gap between every model in
the table. None of these rankings are statistically distinguishable from each other at this
sample size — the paired permutation tests you already have confirm this (all p > 0.38).

**The frozen 13-event secondary split is badly miscalibrated** (threshold -2.53, TN=0, FP=7):
the model flags every single normal test event as anomalous. That is a 13-event validation set
choosing an unusable threshold, not a property of the model.

## What this means for "get the metrics high"

I'm not going to chase a number by changing the evaluation until something clears an arbitrary
bar — that's the thing I said I wouldn't do, and your own D6 control exists specifically to catch
it. But your diagnostics point at something real and fixable: **you already have a working
signal (the classical baseline) that every deep model is failing to match or incorporate.**

The legitimate, fast lever here is not "train harder" (you're overfitting, not underfitting) —
it's **combine what already works.** Two additions, both computed from CSVs you already have on
Drive, neither requiring a GPU or retraining:

1. **Rank ensemble of the deep models**, selected by their *inner-validation* criterion (already
   saved in `seed_stability_inner_validation.csv`) — never by test performance. Averaging
   z-scored out-of-fold scores from multiple imperfect, high-variance models is a standard
   variance-reduction step and is expected to help precisely because your per-model CIs are huge
   relative to the differences between models.
2. **A nested deep+classical blend.** D6 proved the classical signal is real and not leaking.
   The blend weight is chosen per test fold using *only the other four folds'* already-unbiased
   out-of-fold scores — this is exactly how stacked generalization is supposed to work, and it
   cannot see any test fold's own labels when choosing that fold's weight.

I built and unit-tested the exact mechanism in `test_ensemble_blend.py` against synthetic data
shaped like your real layout (58 events, 22 assets, 27 positive, 5 asset-grouped folds) before
writing it into the notebook, and it asserts zero leakage structurally (not just by inspection):

```
ensemble OOF covers every event exactly once: PASS
blended OOF covers every event exactly once: PASS
no test-fold row touched during any fold's selection step: PASS (asserted above)
```

**I have not run this against your real data and do not know what numbers it will produce.**
I don't have your npz cache or GPU-produced OOF scores in this sandbox. What I can tell you is
that on synthetic data with the same shape, the ensemble and blend both improved on every
individual component — which is the expected behavior of variance reduction, not a guarantee
about your specific dataset.

## Also fixed

- The frozen-split threshold instability is a known failure mode of picking a threshold from 13
  events. I've demoted it in the final report and recommend not treating it as a headline number.
- Everything below is additive to your existing saved CSVs — it does not change CELL 1-13, so
  your existing training runs and checkpoints remain valid.

## What I'm not doing

- Not changing `choose_threshold`'s objective to whatever makes numbers look best.
- Not picking which models to ensemble based on their test-fold ranking.
- Not suppressing the classical-baseline comparison, the duration control, or the rank statement.
- Not claiming a result until you've actually run the new cell against your real CSVs.

Run the new cell (instructions below) and send me the printed output — I'll help interpret
whatever it actually says, including if the honest answer is "still doesn't beat the classical
baseline," which would itself be a legitimate and reportable finding for your writeup (it tells
you the deep architectures aren't yet extracting more from the 234 channels than robust quantile
summaries do at n=58).
