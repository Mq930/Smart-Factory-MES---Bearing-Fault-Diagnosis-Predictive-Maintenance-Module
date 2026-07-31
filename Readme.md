## Design notes
- **Group-wise split**: train/val/test are split by source recording (file),
  not by individual window, to prevent leakage between near-duplicate
  overlapping windows. See `group_split()` in `dataset.py`.
- **Model**: 3 parallel Conv1D branches (kernels 8/16/64) -> concat -> 1x1
  conv projection to d_model=128 -> 4-head Transformer encoder (2 layers) ->
  mean pool -> linear classifier. ~372K params, light enough for 8GB VRAM
  with large batch sizes to spare.
- **Grad-CAM hook point**: `model.fused_features` holds the (B, d_model, L)
  post-CNN/pre-Transformer activation map, exposed for the 1D Grad-CAM
  adaptation in the next module.