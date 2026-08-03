# Bearing Fault Diagnosis - Model Layer

## Setup
```bash
pip install torch scipy scikit-learn numpy matplotlib langgraph groq
```

## 1. Download data (run on your own machine, not sandboxed)
```bash
cd src
python download_cwru.py --out ../data/raw
```
Pulls 40 files: Normal baseline (4 loads) + 3 fault types (Inner Race,
Outer Race @6:00, Ball) x 3 severities (0.007"/0.014"/0.021") x 4 loads.
If any file IDs 404, cross-check them against
https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
and update `FILE_MAP` in `download_cwru.py`.

## 2. Train
```bash
# Easier protocol: random file-level split, loads mixed across splits
python train.py --raw_dir ../data/raw --epochs 40 --batch_size 64 --split_strategy group

# Harder, more honest protocol: test on a held-out load the model never trained on
python train.py --raw_dir ../data/raw --epochs 40 --batch_size 64 \
    --split_strategy cross_load --test_load 3

# Add noise augmentation to simulate a real factory floor (train-time only)
python train.py --raw_dir ../data/raw --epochs 40 --batch_size 64 \
    --split_strategy cross_load --test_load 3 --train_noise_std 0.3
```
Saves `../checkpoints/best_model.pt`, `norm_stats.json`, `history.json`.
The checkpoint remembers which split_strategy/test_load/seed it was
trained with, so `evaluate.py` auto-matches without repeating flags.

## 3. Calibrate confidence (recommended before evaluate/MOM agent)
```bash
python calibration.py --checkpoint ../checkpoints/best_model.pt
```
Neural net softmax outputs are often NOT trustworthy probabilities - a
model can say "99% confident" and be wrong far more than 1% of the time.
This fits **temperature scaling** (Guo et al. 2017): a single scalar `T`
optimized on the VALIDATION set (never test) that rescales logits before
softmax. Predictions (argmax) are mathematically unchanged - only the
confidence VALUES become trustworthy. This matters here specifically
because the MOM agent's `route_severity()` uses a hard confidence
threshold to pick `watch` vs `alert`/`critical`.

Saves `calibration.json` (the fitted `T` + before/after ECE) next to the
checkpoint, plus `reliability_before.png` / `reliability_after.png` -
bar charts of predicted confidence vs. actual accuracy per bin (a
perfectly calibrated model's bars sit on the diagonal). Both
`evaluate.py` and the MOM agent's `classifier_tool.py` auto-detect and
apply `calibration.json` if present, falling back to uncalibrated
(T=1.0) with a warning if you skip this step.

## 4. Evaluate
```bash
python evaluate.py --raw_dir ../data/raw --checkpoint ../checkpoints/best_model.pt
```
Prints classification report + confusion matrix on the clean test set,
then runs a **noise-robustness sweep**: re-evaluates the same test
recordings with Gaussian noise injected at increasing severity
(default std levels 0.1/0.3/0.5/0.8, override with `--noise_levels`),
so you get a degradation curve rather than a single number. Applies
calibration.json if present (see step 3). Saves everything to
`test_results.json`.

## 5. Grad-CAM saliency visualization
```bash
python visualize_grad_cam.py --checkpoint ../checkpoints/best_model.pt --n_per_class 2
```
For each class, picks real held-out test windows, runs Grad-CAM, and
saves a PNG showing the raw waveform colored by saliency intensity
plus an isolated saliency curve underneath. Also saves
`grad_cam_summary.json` with one entry per visualized window
(`true_class`, `predicted_class`, `confidence`, `correct`, and the raw
1024-length `cam` array) - this is the structured format the React MES
dashboard will consume to overlay saliency on live telemetry.

Filenames flag misclassifications explicitly
(`ball_007_WRONG_pred_outer_race_007_0.png`) so you can immediately
spot and inspect failure cases alongside correct ones.

Also generates a second plot per window, `*_spectrogram.png`: the same
1D Grad-CAM saliency curve, repainted as a heatmap over a spectrogram
of the signal instead of the raw waveform. This makes periodic
fault-impact structure easier to see at a glance (impacts show up as
regularly-spaced vertical energy stripes). **This is not a separate
"2D Grad-CAM"** - the saliency values are identical to the standard
plot's, just resampled onto the spectrogram's time axis and broadcast
across frequency rows for display. There is no independent
frequency-axis saliency information; describe it as "1D Grad-CAM
saliency overlaid on a spectrogram," not as a distinct 2D method.
Disable with `--no_spectrogram` if you only want the standard plot.

## 6. LangGraph MOM (Manufacturing Operations Management) agent
Lives in `mom_agent/` (separate from `src/`). Requires a Groq API key:
```bash
export GROQ_API_KEY=your_key_here      # Linux/Mac
set GROQ_API_KEY=your_key_here         # Windows cmd
```

```bash
cd mom_agent
python query_cli.py --checkpoint ../checkpoints/best_model.pt \
    --raw_dir ../data/raw --class_name inner_race_014 --load 0
```
On startup, pulls a batch of consecutive windows from a real recording
(simulating a rolling sensor feed - swap `load_windows_from_recording` in
`classifier_tool.py` for a live ingestion source in production), runs the
full agent graph, and prints a diagnosis summary + RCA report. Then drops
into an interactive prompt where you can ask plain-English questions
("should I stop the machine?", "how severe is this?") answered by the LLM
using only the current diagnosis state as context. Type `rerun <class_name>
<load>` to pull a new batch and re-diagnose (e.g. `rerun ball_021 2`), or
`quit` to exit.

**Graph flow** (`graph.py`): `ingest -> aggregate_trend -> route_severity
-> rca_report`
- `ingest`: classifies each window in the batch individually (reuses the
  trained checkpoint + Grad-CAM from earlier modules).
- `aggregate_trend`: summarizes the batch into majority class, agreement
  fraction, and mean confidence - a single noisy window can't trigger an
  alert on its own, only a consistent trend across the batch can.
- `route_severity`: deterministic, inspectable rules (not an LLM call) map
  the aggregated trend to `normal` / `watch` / `alert` / `critical`, so
  alert routing is always auditable and reproducible from the stats.
- `rca_report`: only runs the Groq LLM call when a fault was detected. The
  prompt is grounded in `fault_knowledge.py` (a static knowledge base of
  real bearing fault mechanics - BPFI/BPFO/BSF impact mechanisms, typical
  fault progression, severity-appropriate actions) plus the actual
  aggregated classifier stats. The LLM's job is to structure and phrase a
  report from given facts, not to invent the diagnosis or mechanical
  explanation - this keeps RCA output grounded rather than hallucinated.

## 7. FastAPI backend (for the React dashboard)
Lives in `api/` (separate from `src/` and `mom_agent/`). Wraps the same
model/Grad-CAM/MOM-agent pipeline behind HTTP endpoints for the frontend.
```bash
cd api
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```
The checkpoint and MOM agent graph load ONCE at server startup (via
FastAPI's `lifespan` context manager), not per-request. Override the
checkpoint/data paths with env vars if needed:
`BEARING_CHECKPOINT`, `BEARING_RAW_DIR`, `GROQ_MODEL`.

Endpoints:
- `GET /health` - liveness check
- `GET /recordings` - lists every (class_name, load) recording available
  in `data/raw/`, so the frontend can populate a selector
- `GET /telemetry/{class_name}/{load}?n_windows=10&start_at=0` -
  lightweight endpoint (no LLM call) returning raw waveforms + Grad-CAM
  saliency for the live telemetry + saliency chart
- `POST /diagnose` - runs the full MOM agent graph (body:
  `{class_name, load, n_windows, start_at}`), returns majority class,
  alert level, trend stats, and the RCA report. Also stores the result
  server-side so `/query` can reference it.
- `POST /query` - body `{question}`, answers a natural-language question
  grounded in the most recent `/diagnose` call; returns 400 if
  `/diagnose` hasn't been called yet in the session

CORS is enabled for `localhost:5173`/`localhost:3000` (Vite/CRA dev
server defaults) - adjust `allow_origins` in `main.py` for other setups
or production deployment.

## Design notes
- **Confidence calibration** (`calibration.py`): temperature scaling fits
  a single scalar T on the validation set to correct over/under-confident
  softmax outputs, without changing any prediction. `T` is stored in
  `calibration.json` and auto-applied by both `evaluate.py` and the MOM
  agent's `classifier_tool.py`. Important because `route_severity()` in
  `mom_agent/graph.py` makes hard alert-routing decisions based on a
  confidence threshold - uncalibrated confidence would make that
  threshold's meaning unreliable.
- **10-class problem**: `normal` + {inner_race, outer_race, ball} x
  {007, 014, 021} severities. Much harder than the original 4-class
  single-severity setup, and closer to what a real MES predictive-
  maintenance system needs to distinguish.
- **Two split strategies** (`dataset.py`):
  - `group` - random split by recording file, stratified by class.
    Prevents window-level leakage but loads are mixed across splits.
  - `cross_load` - test set is an entirely unseen operating load.
    The harder, more defensible protocol: separates "did it memorize
    this load's fingerprint" from "did it learn the fault physics".
- **Train-time noise augmentation** (`--train_noise_std`): fresh i.i.d.
  Gaussian noise added to training windows every epoch, in normalized-
  signal units. Val/test always stay clean during training; noise
  robustness is measured separately via `evaluate.py`'s sweep so you
  get both a clean-signal number and a degradation curve.
- **Model**: 3 parallel Conv1D branches (kernels 8/16/64) -> concat -> 1x1
  conv projection to d_model=128 -> 4-head Transformer encoder (2 layers) ->
  mean pool -> linear classifier. ~372K params, light enough for 8GB VRAM
  with large batch sizes to spare.
- **Grad-CAM adaptation for 1D signals** (`grad_cam.py`): hooks
  `model.fused_features`, the (B, d_model, L) post-CNN/pre-Transformer
  activation map. This layer is used instead of anything inside the
  Transformer because self-attention mixes information globally across
  all positions, which would blur time-axis localization - the CNN
  output is the last point in the network with a clean, direct
  correspondence back to specific samples in the raw input. Standard
  Grad-CAM math (gradient-weighted channel combination, ReLU, min-max
  normalize) is applied, then the result is linearly upsampled from L
  back to the full 1024-sample window for overlay on the raw waveform.

