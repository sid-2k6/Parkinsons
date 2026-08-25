# CARE Wind Farm C — Event-Bag Benchmark (V4)

`CARE_WindFarmC_EventBag_V4.ipynb` — a corrected rebuild of the earlier
`CARE_C_NATIVE_EVENT_MSWT_V3` event-level early fault detection benchmark.

## Scope note

This notebook is unrelated to the Parkinson's multimodal work in the rest of this
repository. It lives here because that is where it was requested.

## What this does and does not promise

It **does not guarantee excellent metrics**, and no notebook can. The ceiling is set by
the data: 58 events from 22 turbines, with a single label attached to a prediction frame
that averages ~375 h. What V4 does is remove the mechanisms by which V3 was demonstrably
losing signal, and remove the mechanisms by which it may have been gaining fake signal.
Whether that yields strong numbers is an empirical question the notebook answers
honestly — CELL 22 prints an explicit refusal to claim SOTA when the ranking does not
support it, and CELL 15 prints a duration-only control AUC beside the model AUC.

## The five substantive changes

**1. Objective now matches the reported metric.** V3 trained window-level BCE and scored
events with top-10% pooling, so the loss was pushing windows 300 h before a fault toward
1 — degrading exactly the direction the aggregate depends on. V4 trains at the bag level:

```
bag_logit = mean(top-k window logits),  k/K = 0.10
L = BCE(bag_logit, y_event) + 0.3 * BCE(window_logits, 0) on normal events only
```

The second term exploits an asymmetry V3 ignored: every window in a normal event is a
genuine negative; most windows in an anomaly event are not genuine positives.

**2. Multi-horizon context.** A 10-hour window cannot see a degradation trend. V4 adds
trailing baseline-deviation means at 1 d / 7 d / 30 d per channel through a small MLP
branch. All six main-comparison models receive it, so the proposed model has no
structural advantage; `use_ctx=False` is ablation rung A4, not a handicap on baselines.

**3. Channel hygiene.** V3's `scale` fell back to `1.0` for channels that were
near-constant in the baseline year, so values saturated at ±12 — and *which* channels
saturated differed per event, acting as a turbine fingerprint. V4 clips at ±8 and drops
channels that are degenerate in >20% of events, behave as cumulative counters
(`|corr(value, t)| > 0.90`), or whose clipping rate alone predicts the label
(`AUC > 0.75`). The report is written to `channel_hygiene.csv`.

**4. Selection no longer runs on noise.** V3 early-stopped on validation event PR-AUC over
6 positives — a statistic with a handful of discrete values — with `PATIENCE=3` and
`MAX_EPOCHS=12`, so most runs stopped mid-anneal. V4 uses
`0.7 * event ROC-AUC + 0.3 * window ROC-AUC` on inner validation, plus EMA weights,
warmup + full cosine, 40 epochs, patience 8.

**5. Statistical power.** V3 reported 13 test events (6 positive), where recall is
quantised at 1/6. V4's primary evaluation is 5-fold asset-grouped nested CV over all 58
events (27 positive), with the operating threshold taken from each fold's own inner
validation and predictions pooled. The frozen 32/13/13 split is kept as a secondary
result. Confidence intervals use a cluster bootstrap over **assets**, because events from
one turbine are not independent, plus an asset-level paired permutation test.

## Bugs fixed

| Bug | Effect |
|---|---|
| `st = max(0, min_ep - LOOKBACK + 1)` with no endpoint guard | negative slice start → assertion crash mid-run |
| `torch.load` without `weights_only=True` | raises on torch ≥ 2.6, breaking checkpoint resume |
| `nextafter(min, -inf)` in the threshold candidate set | "predict everything positive" could win, giving recall 1.0 / precision 0.46 |
| `num_workers=2` over an in-RAM `records` list | forked copies of the dataset per worker |
| bootstrap resampling events from 4 test assets | anti-conservative confidence intervals |
| figure script pasted twice, first copy truncated at `zorder=(10` | `SyntaxError` |
| `TN=5 FP=2 FN=1 TP=5` and lead-time percentages typed by hand | figures silently contradict the CSVs on rerun |
| `.max()` of the rolling risk | length bias — anomaly events average 375.7 h vs 318.8 h |
| false alarms counted per event | replaced by alarms per 100 normal-event days |

Six trailing cells were removed: two duplicate figure scripts, two duplicate confusion
matrices, a checkpoint-existence scan, and a truncated window-level inference cell.

## Structure

| Cells | Purpose |
|---|---|
| 1–4 | protocol constants, Drive mount, schema verification, frozen split + asset folds |
| 5, 5B | cache with hygiene flags and 1/7/30-day context; global channel hygiene mask |
| 6 | GPU-resident event store (replaces `WindowDataset` / `DataLoader`) |
| 6D | diagnostics D3/D5/D6 — run once before training |
| 8–12 | scoring, backbones, baselines, MIL trainer, model registry |
| 13–14 | asset-grouped nested CV, pooled metrics, bootstrap, permutation tests |
| 15–20 | figures and the operational rolling-6h lead-time analysis |
| 21–22 | reproducibility outputs and final report |

There are three checkpoint markdown cells. They are worth stopping at, particularly
Checkpoint 2 (`KEPT n / 238`) and Checkpoint 3 (diagnostics).

## Runtime

Full grid is ~168 trainings, roughly 8–19 h on a T4. Defaults reduce this to ~108:
`REUSE_A0_A3 = True` (A0/A3 are architecturally identical to Plain Transformer and the
proposed model) and `ABLATION_SEEDS = [42]`. Set `RESUME_FROM_CHECKPOINTS = True` for runs
after the first — but only while CELL 1 is unchanged, since stale checkpoints load
silently. Time one fold before committing:

```python
import time
t0 = time.time()
run_grouped_cv(PROPOSED, MAIN_MODELS[PROPOSED], FOLDS[:1], seeds=[42])
print((time.time() - t0) / 60, "min")
```

## Before the first run

Delete or rename any existing `cache_CARE_C_EVENT_MIL_V4` directory. The npz layout gained
a `ctx` key and stale files raise `KeyError`.

## Two things to verify while running

**CELL 5B** — if `drop_sat_leak` is non-empty, that is direct evidence the old ±12 clipping
was leaking the label. Report it. If fewer than ~150 channels survive, check
`channel_hygiene.csv` against `feature_description.csv` and raise `RAMP_TOL` rather than
discarding real sensors.

**CELL 20** — the cell prints event-frame duration by class before anything else.
`lead_hours` is `event_end − first_alarm`. If `event_end` is only the end of the CSV
prediction section rather than the recorded fault time, every lead time is inflated by the
post-fault tail and the "83.3% at ≥6 h" claim must be restated.

## Honesty constraints

- Fix every choice — architecture, epochs, threshold, aggregation, hygiene tolerances —
  from inner validation folds only. Run the design once and report what it says. Tuning
  against the pooled 58-event numbers invalidates the confidence intervals and any SOTA
  statement.
- The independent test units are **events**: 58 out-of-fold, 27 positive. Window counts are
  repeated overlapping observations of those events and must never be reported as a sample
  size. Keep the three figures separate in writing: 58 events, the window count, and the
  abandoned 7,913-window strict-6 h formulation.
- Pooled PR-AUC and ROC-AUC combine scores from five different fold models. Report them as
  a pooled ranking summary of out-of-fold discrimination, not as a calibrated single-model
  AUC. The threshold metrics are exact.
- The named modern architectures are task adaptations under one common CARE protocol, not
  byte-identical reproductions of the original repositories.

## Verification status

Static only: valid nbformat, all 23 code cells parse, no undefined names and no
used-before-defined names across cells. It has **not** been executed — that requires the
CARE Wind Farm C data on Google Drive.
