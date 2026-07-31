"""
Multi-Scale 1D-CNN + Transformer Encoder for bearing fault diagnosis.

Architecture:
    Input: (B, 1, 1024) raw vibration window
      -> 3 parallel Conv1D branches (kernel sizes 8, 16, 64), each producing
         (B, C_branch, L) feature maps that capture fault signatures at
         different frequency/time scales (small kernel = high-frequency
         transients like ball/inner-race impacts; large kernel = broader
         periodic patterns like outer-race passage).
      -> Concatenate branch features along channel dim -> (B, 3*C_branch, L)
      -> Project to d_model, treat L positions as a sequence
      -> Add learnable positional embedding
      -> 4-head Transformer encoder (configurable depth)
      -> Mean-pool over sequence -> (B, d_model)
      -> Classification head -> (B, num_classes)

Grad-CAM compatibility:
    The model exposes `self.fused_features` (post-CNN, pre-Transformer,
    shape (B, d_model, L)) as the target activation map for Grad-CAM,
    since that's the representation with a direct, interpretable time-axis
    correspondence back to the input signal. See grad_cam.py.
"""

import math
import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    """One scale of the multi-scale CNN front-end: Conv1D -> BN -> ReLU,
    stacked twice, with stride-2 downsampling to control sequence length."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=2, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)  # (B, out_channels, L/4)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1), :]


class MultiScaleCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        branch_channels: int = 32,
        kernel_sizes=(8, 16, 64),
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.branches = nn.ModuleList(
            [ConvBranch(in_channels, branch_channels, k) for k in kernel_sizes]
        )
        fused_channels = branch_channels * len(kernel_sizes)

        # project fused multi-scale features to transformer dimension
        self.input_proj = nn.Sequential(
            nn.Conv1d(fused_channels, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # populated on forward() for Grad-CAM use
        self.fused_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, window_size) raw vibration window
        returns: (B, num_classes) logits
        """
        branch_outs = [branch(x) for branch in self.branches]

        # branches use the same stride/padding scheme so lengths match;
        # trim defensively to the shortest branch just in case of off-by-one
        min_len = min(b.shape[-1] for b in branch_outs)
        branch_outs = [b[..., :min_len] for b in branch_outs]

        fused = torch.cat(branch_outs, dim=1)  # (B, fused_channels, L)
        fused = self.input_proj(fused)         # (B, d_model, L)

        # keep for Grad-CAM (detach not applied here - hook/backward needs graph)
        self.fused_features = fused

        seq = fused.transpose(1, 2)            # (B, L, d_model)
        seq = self.pos_encoding(seq)
        seq = self.transformer(seq)            # (B, L, d_model)

        pooled = seq.mean(dim=1)               # (B, d_model)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def build_model(num_classes: int = 4, **kwargs) -> MultiScaleCNNTransformer:
    return MultiScaleCNNTransformer(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Quick shape/sanity test
    model = build_model()
    dummy = torch.randn(8, 1, 1024)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (8, 4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
