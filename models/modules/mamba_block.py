import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiDirectionalMamba(nn.Module):
    """Bidirectional Mamba processing block."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        # Forward Mamba.
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if self.bidirectional:
            # Backward Mamba.
            self.mamba_bwd = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def forward(self, x):
        # x shape: (B, L, C), where L is the point-sequence length.
        
        # 1. Forward scan.
        fwd_out = self.mamba_fwd(x)
        if not self.bidirectional:
            return fwd_out
        
        # 2. Flip the sequence, scan backward, then flip back.
        x_flipped = torch.flip(x, dims=[1])
        bwd_out = self.mamba_bwd(x_flipped)
        bwd_out = torch.flip(bwd_out, dims=[1])
        
        # 3. Fuse bidirectional features by summation to save memory.
        return fwd_out + bwd_out


class PointMambaBlock(nn.Module):
    """
    Spatially aware Point Mamba block.
    Pipeline: coordinate sorting -> serialization -> bidirectional Mamba
    -> inverse sorting.
    """
    def __init__(self, d_model, axes=("x", "y", "z"), bidirectional=True, restore_order=True):
        super().__init__()
        self.d_model = d_model
        self.axes = tuple(axes)
        self.restore_order = restore_order
        if not self.axes:
            raise ValueError("PointMambaBlock requires at least one axis.")
        
        self.axis_to_dim = {"x": 0, "y": 1, "z": 2}
        for axis in self.axes:
            if axis not in self.axis_to_dim:
                raise ValueError(f"Unsupported Mamba sort axis: {axis}")

        # Separate Mamba modules for the enabled axes.
        self.mambas = nn.ModuleDict({
            axis: BiDirectionalMamba(d_model, bidirectional=bidirectional)
            for axis in self.axes
        })
        
        # Linear projection for feature aggregation.
        self.proj = nn.Linear(d_model * len(self.axes), d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        
        # Zero-initialize to preserve the pretrained LFA-Mamba feature flow.
        nn.init.constant_(self.proj.weight, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, xyz, features):
        """
        Args:
            xyz: point coordinates (B, N, 3), used as the sorting prior.
            features: point features (B, N, C), used as the input sequence.
        Returns:
            Processed point features (B, N, C).
        """
        B, N, C = features.shape
        
        # Expand indices to match the feature tensor rank for gather.
        # dummy_idx shape: (B, N, C)
        def gather_features(feat, idx):
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)
            return torch.gather(feat, dim=1, index=idx_expanded)

        outputs = []
        for axis in self.axes:
            idx = torch.argsort(xyz[:, :, self.axis_to_dim[axis]], dim=-1)
            feat_sorted = gather_features(features, idx)
            out = self.mambas[axis](feat_sorted)
            if self.restore_order:
                idx_inv = torch.argsort(idx, dim=-1)
                out = gather_features(out, idx_inv)
            outputs.append(out)

        # Feature fusion with residual connection.
        merged_feat = torch.cat(outputs, dim=-1)
        
        # Project back to C channels and add the original features.
        out = self.proj(merged_feat)
        out = self.act(self.norm(out))
        
        return features + out
