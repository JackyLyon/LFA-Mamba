import torch


def _min_distances(src, dst, chunk_size=4096):
    if src.numel() == 0 or dst.numel() == 0:
        return torch.empty(src.shape[0], device=src.device, dtype=src.dtype)

    mins = []
    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        dist = torch.cdist(src[start:end].unsqueeze(0), dst.unsqueeze(0), p=2).squeeze(0)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins, dim=0)


def _nearest_indices(src, dst, chunk_size=4096):
    if src.numel() == 0 or dst.numel() == 0:
        return torch.empty(src.shape[0], device=src.device, dtype=torch.long)

    indices = []
    for start in range(0, src.shape[0], chunk_size):
        end = min(start + chunk_size, src.shape[0])
        dist = torch.cdist(src[start:end].unsqueeze(0), dst.unsqueeze(0), p=2).squeeze(0)
        indices.append(dist.min(dim=1).indices)
    return torch.cat(indices, dim=0)


def _otsu_threshold(values, bins=128):
    values = values.detach()
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)

    min_value = values.min()
    max_value = values.max()
    if torch.isclose(min_value, max_value):
        return max_value

    hist = torch.histc(values, bins=bins, min=float(min_value), max=float(max_value))
    bin_centers = torch.linspace(min_value, max_value, bins, device=values.device, dtype=values.dtype)
    prob = hist.to(values.dtype)
    prob = prob / prob.sum().clamp_min(1e-12)

    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * bin_centers, dim=0)
    mu_total = mu[-1]
    between = (mu_total * omega - mu).pow(2) / (omega * (1.0 - omega)).clamp_min(1e-12)
    between = torch.nan_to_num(between, nan=0.0, posinf=0.0, neginf=0.0)
    return bin_centers[between.argmax()]


def _infer_defect_mask(gt, partial):
    dist_to_partial = _min_distances(gt, partial)
    threshold = _otsu_threshold(dist_to_partial)
    defect_mask = dist_to_partial > threshold

    min_points = max(8, int(gt.shape[0] * 0.01))
    if defect_mask.sum() < min_points:
        threshold = torch.quantile(dist_to_partial, 0.90)
        defect_mask = dist_to_partial > threshold
    if (~defect_mask).sum() < min_points:
        threshold = torch.quantile(dist_to_partial, 0.10)
        defect_mask = dist_to_partial > threshold

    return defect_mask, threshold


def _infer_boundary_mask(gt, defect_mask, threshold):
    observed = gt[~defect_mask]
    defect = gt[defect_mask]
    if observed.numel() == 0 or defect.numel() == 0:
        return torch.ones(gt.shape[0], device=gt.device, dtype=torch.bool)

    radius = torch.clamp(threshold * 2.0, min=1e-6)
    dist_to_defect = _min_distances(gt, defect)
    dist_to_observed = _min_distances(gt, observed)
    boundary_mask = ((~defect_mask) & (dist_to_defect <= radius)) | (defect_mask & (dist_to_observed <= radius))

    min_points = max(8, int(gt.shape[0] * 0.01))
    if boundary_mask.sum() < min_points:
        combined = torch.minimum(dist_to_defect, dist_to_observed)
        cutoff = torch.quantile(combined, 0.10)
        boundary_mask = combined <= cutoff

    return boundary_mask


def _region_chamfer(pred, gt, region_mask):
    gt_region = gt[region_mask]
    if gt_region.numel() == 0:
        gt_region = gt
        region_mask = torch.ones(gt.shape[0], device=gt.device, dtype=torch.bool)

    gt_to_pred = _min_distances(gt_region, pred).mean()

    nearest_gt_idx = _nearest_indices(pred, gt)
    pred_region = pred[region_mask[nearest_gt_idx]]
    if pred_region.numel() == 0:
        pred_region = pred

    pred_to_gt = _min_distances(pred_region, gt_region).mean()
    return gt_to_pred + pred_to_gt


def get_defect_boundary_cd(pred, gt, partial):
    """Compute defect-aware Chamfer metrics.

    Args:
        pred: [B, N, 3] final prediction.
        gt: [B, M, 3] complete ground truth.
        partial: [B, K, 3] observed partial input.

    Returns:
        Tuple of Python floats: (CD-defect * 1e3, CD-boundary * 1e3).
    """
    cd_defect = []
    cd_boundary = []
    for i in range(pred.shape[0]):
        defect_mask, threshold = _infer_defect_mask(gt[i], partial[i])
        boundary_mask = _infer_boundary_mask(gt[i], defect_mask, threshold)
        cd_defect.append(_region_chamfer(pred[i], gt[i], defect_mask))
        cd_boundary.append(_region_chamfer(pred[i], gt[i], boundary_mask))

    return (
        torch.stack(cd_defect).mean().item() * 1e3,
        torch.stack(cd_boundary).mean().item() * 1e3,
    )
