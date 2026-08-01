## Design notes
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