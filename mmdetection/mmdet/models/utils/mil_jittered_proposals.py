# Copyright (c) OpenMMLab. All rights reserved.
"""Shared jittered proposal generation for MIL train/infer alignment."""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch


def generate_jittered_proposals(
    points: torch.Tensor,
    img_shape: Union[torch.Tensor, Sequence[float]],
    base_scales: Sequence[Union[int, float]],
    ratios: Sequence[Union[int, float]],
    anchor_offsets: Sequence[Tuple[float, float]],
    min_side: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-scale multi-offset boxes around each point (aligned with legacy loop).

    Args:
        points: Tensor of shape (N, 2) as (x, y) in the same coordinate frame as
            ``img_shape`` (typically ``img_meta['img_shape']`` for input-space points).
        img_shape: (h, w) image size for clipping.
        base_scales: Base scales (same role as ``infer_base_scales``).
        ratios: Aspect ratios (same role as ``infer_ratios``).
        anchor_offsets: (dx, dy) offsets in units of box width/height respectively.
        min_side: Proposals with width or height <= this value are dropped (strict
            inequality matches ``> min_side`` in the reference loop).

    Returns:
        proposals: (M, 4) xyxy on ``points.device`` / dtype.
        point_indices: (M,) long, index of the source point per proposal.
    """
    if points.numel() == 0:
        dev = points.device
        dt = points.dtype
        return (
            torch.empty((0, 4), device=dev, dtype=dt),
            torch.empty((0,), device=dev, dtype=torch.long),
        )

    device = points.device
    dtype = points.dtype
    if isinstance(img_shape, torch.Tensor):
        h = float(img_shape[0].item())
        w = float(img_shape[1].item())
    else:
        h = float(img_shape[0])
        w = float(img_shape[1])

    scales_t = torch.as_tensor(base_scales, dtype=dtype, device=device).reshape(-1)
    ratios_t = torch.as_tensor(ratios, dtype=dtype, device=device).reshape(-1)
    if scales_t.numel() == 0 or ratios_t.numel() == 0:
        return (
            torch.empty((0, 4), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    sqrt_r = torch.sqrt(ratios_t)
    h_box = scales_t[:, None] * sqrt_r[None, :]  # (S, R)
    w_box = scales_t[:, None] / sqrt_r[None, :]

    off_list: List[tuple] = [tuple(o) for o in anchor_offsets]
    if len(off_list) == 0:
        return (
            torch.empty((0, 4), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )
    off_t = torch.as_tensor(off_list, dtype=dtype, device=device)  # (O, 2)

    N = points.size(0)
    S, R = h_box.shape
    O = off_t.size(0)

    px = points[:, 0].view(N, 1, 1, 1)
    py = points[:, 1].view(N, 1, 1, 1)
    wb = w_box.view(1, S, R, 1)
    hb = h_box.view(1, S, R, 1)
    ox = off_t[:, 0].view(1, 1, 1, O)
    oy = off_t[:, 1].view(1, 1, 1, O)

    cx = px + ox * wb
    cy = py + oy * hb
    x1 = cx - wb * 0.5
    y1 = cy - hb * 0.5
    x2 = cx + wb * 0.5
    y2 = cy + hb * 0.5

    x1 = x1.clamp(min=0.0, max=w)
    y1 = y1.clamp(min=0.0, max=h)
    x2 = x2.clamp(min=0.0, max=w)
    y2 = y2.clamp(min=0.0, max=h)

    # Element-wise combine masks; Python `and` on tensors is invalid (ambiguous bool).
    valid = ((x2 - x1) > min_side) & ((y2 - y1) > min_side)
    if not valid.any():
        return (
            torch.empty((0, 4), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    n_idx = torch.arange(N, device=device).view(N, 1, 1, 1).expand_as(x1)
    x1f = x1[valid]
    y1f = y1[valid]
    x2f = x2[valid]
    y2f = y2[valid]
    point_indices = n_idx[valid]

    proposals = torch.stack([x1f, y1f, x2f, y2f], dim=-1)
    return proposals, point_indices.reshape(-1).long()
