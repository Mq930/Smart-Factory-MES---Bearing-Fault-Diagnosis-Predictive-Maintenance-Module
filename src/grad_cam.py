"""
Grad-CAM adapted for 1D vibration signals.

Standard Grad-CAM (Selvaraju et al.) was designed for 2D image CNNs, where
the target layer is the last convolutional feature map before global
pooling. Here we adapt the same idea to a 1D CNN + Transformer:

    - Target layer: `model.fused_features`, the (B, d_model, L) activation
      map produced by the multi-scale CNN front-end, right before it's fed
      into the Transformer encoder. We hook this layer (not something
      inside the Transformer) because:
        1. It has a direct, well-defined receptive field back to specific
           samples in the raw 1024-length input (each of the L positions
           corresponds to a ~4-sample chunk of the original signal, from
           the two stride-2 convs in each CNN branch).
        2. Transformer self-attention mixes information globally across
           all L positions, so gradients at a post-Transformer layer no
           longer localize to specific time segments in the same way -
           using a post-Transformer layer would give a much blurrier,
           less physically meaningful saliency map.

    - Importance weights: global-average-pool the gradient of the target
      class logit w.r.t. fused_features over the time dimension L, giving
      one scalar weight per channel (exactly like 2D Grad-CAM pools over
      H x W instead of just L).

    - CAM: weighted sum of the (unpooled) activation channels, followed by
      ReLU (keep only features that positively support the predicted
      class) and min-max normalization to [0, 1].

    - Upsampling: the raw CAM has length L (~64 for a 1024-sample input
      after two stride-2 convs per branch); we linearly interpolate it
      back to the original 1024 samples so it can be overlaid directly on
      the input waveform.

Usage:
    from grad_cam import GradCAM1D
    cam_tool = GradCAM1D(model)
    cam, pred_class, probs = cam_tool.generate(x)   # x: (1, 1, window_size)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM1D:
    """
    Wraps a MultiScaleCNNTransformer (or any model exposing a
    `self.fused_features` tensor of shape (B, C, L) set during forward())
    and computes Grad-CAM saliency over the original input length.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def generate(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        input_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Args:
            x: input tensor, shape (1, 1, window_size). Batch size must be 1 -
               Grad-CAM here is computed per-example so the backward pass
               targets a single scalar logit.
            target_class: which class's logit to explain. If None, uses the
               model's own predicted (argmax) class - i.e. "why did the
               model think this is class X", which is the common case.
               Pass an explicit index to ask "what would make this LOOK
               like class Y" instead (useful for contrastive explanations).
            input_length: original input length to upsample the CAM to.
               Defaults to x.shape[-1].

        Returns:
            cam: 1D numpy array of length input_length, values in [0, 1].
                 Higher = more influential for the target class.
            target_class: the class index actually explained (resolved from
                 argmax if None was passed).
            probs: (num_classes,) softmax probabilities from the forward pass,
                 for reporting confidence alongside the saliency map.
        """
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected x of shape (1, 1, window_size), got {tuple(x.shape)}")

        input_length = input_length or x.shape[-1]

        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(False)  # input grad not needed, only fused_features grad

        logits = self.model(x)  # (1, num_classes); also populates model.fused_features
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        fused = self.model.fused_features  # (1, d_model, L), has grad_fn
        if fused is None or not fused.requires_grad:
            raise RuntimeError(
                "model.fused_features is None or has no grad. Make sure the "
                "model is not wrapped in torch.no_grad() when calling generate()."
            )

        score = logits[0, target_class]
        # retain_graph not needed - single backward per call
        grads = torch.autograd.grad(score, fused, retain_graph=False)[0]  # (1, d_model, L)

        # channel importance = global-average-pool gradients over time (L)
        weights = grads.mean(dim=2, keepdim=True)  # (1, d_model, 1)

        # weighted combination of activation channels
        cam = (weights * fused).sum(dim=1)  # (1, L)
        cam = F.relu(cam)

        # upsample from L back to the original input length
        cam = cam.unsqueeze(1)  # (1, 1, L) for interpolate
        cam = F.interpolate(cam, size=input_length, mode="linear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()  # (input_length,)

        # normalize to [0, 1] for visualization; guard against a flat/all-zero CAM
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, probs


if __name__ == "__main__":
    # Quick smoke test with a randomly initialized model
    import sys
    sys.path.insert(0, ".")
    from model import build_model

    model = build_model(num_classes=10)
    cam_tool = GradCAM1D(model)

    dummy_x = torch.randn(1, 1, 1024)
    cam, pred_class, probs = cam_tool.generate(dummy_x)

    print("CAM shape:", cam.shape, "min/max:", cam.min(), cam.max())
    print("Predicted class:", pred_class)
    print("Probs shape:", probs.shape, "sums to:", probs.sum())
