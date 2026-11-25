import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from models.modules.utils import MLP_Res, MLP_CONV, query_knn, grouping_operation


class TangentNormalAdaptiveUpsampling(nn.Module):
    """
    Local-frame Adaptive Up-sampling Module.
    Decomposes offsets into tangent-plane directions for surface expansion
    and normal directions for detail refinement. The network also predicts
    anisotropic scale factors, with Gram-Schmidt stabilization.
    """
    def __init__(
        self,
        up_factor=4,
        use_local_frame=True,
        use_tangent_normal=True,
        use_anisotropic_scale=True,
        use_gate=True,
    ):
        super().__init__()
        self.up_factor = up_factor
        self.use_local_frame = use_local_frame
        self.use_tangent_normal = use_tangent_normal
        self.use_anisotropic_scale = use_anisotropic_scale
        self.use_gate = use_gate

        # Keep the original DeConv path.
        self.decrease_dim = MLP_CONV(in_channel=128, layer_dims=[64, 32], bn=True)
        self.ps      = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)
        self.mlp_res = MLP_Res(in_dim=128 * 2, hidden_dim=128, out_dim=128)
        self.upper   = nn.Upsample(scale_factor=up_factor)
        self.geo_proj = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.geo_fuse = nn.Sequential(
            nn.Conv1d(128 + 32, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.direct_delta_mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

        # Tangential offsets: two coefficients along local e1/e2.
        self.tangent_mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 2, 1)
        )
        # Normal offset: one coefficient along local e3.
        self.normal_mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 1, 1)
        )
        self.coeff_mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        # Local frame prediction: 9D -> 3x3, then Gram-Schmidt.
        self.frame_mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 9, 1)
        )
        # Anisotropic scale: sigmoid output mapped to [0.5, 2.0].
        self.scale_predictor = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 3, 1), nn.Sigmoid()
        )
        # Start geometry offsets with a small gate to protect early training.
        self.delta_gate = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _gram_schmidt(frame):
        """
        frame: (B, 3, 3, N) -> orthonormalized (B, 3, 3, N).
        Includes zero-vector detection, deterministic fallback, normalize eps,
        and nan_to_num cleanup.
        """
        # e1
        e1 = F.normalize(frame[:, :, 0, :], dim=1, eps=1e-8)

        # Orthogonalize e2, then replace zero vectors with a deterministic fallback.
        e2 = frame[:, :, 1, :]
        e2 = e2 - (e1 * e2).sum(dim=1, keepdim=True) * e1
        e2_norm = e2.norm(dim=1, keepdim=True)

        # Choose a fixed axis not parallel to e1, then project into the tangent plane.
        x_axis = torch.zeros_like(e1)
        x_axis[:, 0, :] = 1.0
        y_axis = torch.zeros_like(e1)
        y_axis[:, 1, :] = 1.0
        use_x = torch.abs(e1[:, 0:1, :]) < 0.9
        fallback_axis = torch.where(use_x, x_axis, y_axis)
        fallback_e2 = fallback_axis - (fallback_axis * e1).sum(dim=1, keepdim=True) * e1
        fallback_e2 = F.normalize(fallback_e2, dim=1, eps=1e-8)

        e2_safe = torch.where(e2_norm < 1e-6, fallback_e2, e2)
        e2 = F.normalize(e2_safe, dim=1, eps=1e-8)

        # Normalize e3 after the cross product.
        e3 = torch.cross(e1, e2, dim=1)
        e3 = F.normalize(e3, dim=1, eps=1e-8)

        result = torch.stack([e1, e2, e3], dim=2)  # (B, 3, 3, N)
        result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        return result

    def forward(self, xyz, feat, geo_feat=None):
        """
        xyz:      (B, 3, N)
        feat:     (B, 128, N)
        geo_feat: (B, 64, N)
        """
        B, _, N = xyz.shape

        feat_child = self.ps(self.decrease_dim(feat))       # (B, 128, N*r)
        feat_child = self.mlp_res(
            torch.cat([feat_child, self.upper(feat)], dim=1)
        )
        base_xyz = self.upper(xyz)                          # (B, 3, N*r)
        N_up = N * self.up_factor

        if geo_feat is not None:
            if geo_feat.shape[-1] != N_up:
                geo_feat = F.interpolate(
                    geo_feat, size=N_up, mode='linear', align_corners=False)
            geo_feat = self.geo_proj(geo_feat)
            feat_child = self.geo_fuse(torch.cat([feat_child, geo_feat], dim=1))

        if not self.use_local_frame:
            delta = torch.tanh(self.direct_delta_mlp(feat_child)) * 0.1
            return base_xyz + delta

        # Local frame.
        frame_raw = self.frame_mlp(feat_child).reshape(B, 3, 3, N_up)
        frame     = self._gram_schmidt(frame_raw)           # (B, 3, 3, N_up)

        if self.use_tangent_normal:
            # Offset coefficients.
            t_coeff = self.tangent_mlp(feat_child)              # (B, 2, N_up)
            n_coeff = torch.tanh(self.normal_mlp(feat_child))   # (B, 1, N_up)
            coeff   = torch.cat([t_coeff, n_coeff], dim=1)      # (B, 3, N_up)
        else:
            coeff = self.coeff_mlp(feat_child)

        # Anisotropic scale in [0.5, 2.0].
        if self.use_anisotropic_scale:
            scale = 0.5 + 1.5 * self.scale_predictor(feat_child)
        else:
            scale = 1.0

        # delta = sum_j coeff_j * frame_j, then apply the scale factor.
        delta = einsum('b j n, b i j n -> b i n', coeff, frame)
        delta = torch.tanh(delta) * scale * 0.1
        if self.use_gate:
            delta = torch.sigmoid(self.delta_gate) * delta

        return base_xyz + delta


TNA = TangentNormalAdaptiveUpsampling


class UpBlcok_TNA(nn.Module):
    def __init__(self, feat_dim=512, down_rates=[1, 4, 2],
                 knns=[16, 12, 8], up_factor=4, tna_config=None):
        super().__init__()
        tna_config = tna_config or {}
        from models.lfa_mamba import CRT, PN
        self.pn        = PN(feat_dim=feat_dim)
        self.inter_crt = CRT(dim_in=128, is_inter=True,
                             down_rates=down_rates, knns=knns)
        self.intra_crt = CRT(dim_in=128, is_inter=False,
                             down_rates=down_rates, knns=knns)
        self.deconv    = TangentNormalAdaptiveUpsampling(up_factor=up_factor, **tna_config)

    def forward(self, p, gf, pre, fps_idxs_1, fps_idxs_2, geo_feat=None):
        h = self.pn(p, gf)
        g, fps_idxs_q1, fps_idxs_s1 = self.inter_crt(
            [p, h], pre, None, fps_idxs_1)
        f, _, _ = self.intra_crt([p, g], [p, g], fps_idxs_q1, fps_idxs_q1)
        new_xyz = self.deconv(p, f, geo_feat=geo_feat)
        return new_xyz, f, fps_idxs_q1, fps_idxs_s1


UpBlcok_TNAU = UpBlcok_TNA


class TNADecoder(nn.Module):
    def __init__(self, tna_config=None):
        super().__init__()
        tna_config = tna_config or {}
        self.ub0 = UpBlcok_TNA(feat_dim=512, down_rates=[1, 2, 2],
                                knns=[16, 12, 8], up_factor=1, tna_config=tna_config)
        self.ub1 = UpBlcok_TNA(feat_dim=512, down_rates=[1, 2, 2],
                                knns=[16, 12, 8], up_factor=4, tna_config=tna_config)
        self.ub2 = UpBlcok_TNA(feat_dim=512, down_rates=[1, 4, 2],
                                knns=[16, 12, 8], up_factor=8, tna_config=tna_config)

    def forward(self, global_f, p0, p_sd, f_sd, geo_feat=None):
        p1, f0, idxs_122, _ = self.ub0(
            p0, global_f, [p_sd, f_sd], None, None, geo_feat=geo_feat)
        p2, f1, idxs_p1, _ = self.ub1(
            p1, global_f, [p0, f0], None, idxs_122, geo_feat=geo_feat)
        p3, _, _, _ = self.ub2(
            p2, global_f, [p1, f1], None, None, geo_feat=geo_feat)

        return [
            p_sd.permute(0, 2, 1).contiguous(),
            p1.permute(0, 2, 1).contiguous(),
            p2.permute(0, 2, 1).contiguous(),
            p3.permute(0, 2, 1).contiguous(),
        ]
