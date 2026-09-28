"""
EdgeDepth-8-dense -- exact dense-1x1 copy of the EdgeDepth-8-prered matching path.

Why this exists
--------------
Hailo-8 cannot place the grouped 1x1 conv (groups = D*G) of the
prered cost volume (``PreReducedCostVolume.fused_corr`` in
src/model_prered.py: Conv2d(D*C, D*G, 1, groups=D*G, bias=False), D=24,
C=32, G=8 -> 768 in, 192 out). Dense convs map well, so this module
replaces that single op with a DENSE Conv2d(768, 192, 1, groups=1,
bias=False) whose weight is the exact block-diagonal expansion of the
grouped weight: for output channel o (group o), input channels
o*k .. o*k+k-1 (k = C//G = 4) carry the grouped weights, all other
entries are 0. Everything else is identical to src/model_prered.py
(imported and reused, never modified).

Math: a grouped 1x1 with groups=N, Cin=N*k, Cout=N (one out per group)
computes y[o] = sum_{j<k} W[o, j] * x[o*k + j]. The dense conv with
weight D[o, i] = W[o, i - o*k] for i in [o*k, o*k+k) and 0 elsewhere
computes exactly the same y.

No retraining: convert_from_prered / convert_from_original only remap weights.
"""

from __future__ import annotations

import copy
import sys
import pathlib

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, _shift_all, NUM_DISP  # noqa: E402
# Reuse the prered/grouped helpers so the channel-layout contract cannot drift.
from model_prered import (  # noqa: E402
    HailoStereoPrered,
    PreReducedCostVolume,
    convert_from_original as _prered_from_original,
)
from model_grouped import (  # noqa: E402
    _stem_channel_perm,
    load_original,
    strip_module_prefix,
)

__all__ = [
    "DenseCorrCostVolume",
    "HailoStereoDense",
    "expand_grouped_to_dense",
    "convert_from_prered",
    "convert_from_original",
    "_stem_channel_perm",
    "load_original",
    "strip_module_prefix",
]


def expand_grouped_to_dense(grouped_weight: torch.Tensor) -> torch.Tensor:
    """Block-diagonal expansion of a grouped 1x1 weight to a dense 1x1 weight.

    Args:
        grouped_weight: [N, k, 1, 1] grouped conv weight with groups=N
            (N out channels, k = Cin/N in channels per group).

    Returns:
        [N, N*k, 1, 1] dense weight with groups=1 such that
        dense[o, o*k:(o+1)*k] == grouped[o, :] and 0 elsewhere.
    """
    assert grouped_weight.dim() == 4
    assert grouped_weight.shape[2] == 1 and grouped_weight.shape[3] == 1
    n, k, _, _ = grouped_weight.shape
    dense = torch.zeros(n, n * k, 1, 1,
                        dtype=grouped_weight.dtype, device=grouped_weight.device)
    for o in range(n):
        dense[o, o * k:(o + 1) * k, 0, 0] = grouped_weight[o, :, 0, 0]
    return dense


class DenseCorrCostVolume(nn.Module):
    """Drop-in copy of PreReducedCostVolume with a dense corr 1x1.

    Only the ``"both"`` cost is supported -- the EdgeDepth-8 arch. Output
    layout is ``torch.cat([sub, corr], 1)`` = ``[all sub (d-major) | all
    corr (d-major)]``, exactly like PreReducedCostVolume, so the same
    stem input-channel permutation applies.
    """

    def __init__(self, ch=32, num_disp=NUM_DISP, groups=8):
        super().__init__()
        assert ch % groups == 0, "channels must divide evenly into groups"
        self.ch = ch
        self.num_disp = num_disp
        self.groups = groups
        self.cost = "both"
        self.reduce = nn.Conv2d(ch, groups, 1, groups=groups, bias=False)
        # Dense twin of PreReducedCostVolume.fused_corr: same in/out
        # channels, groups=1, block-diagonal weight (see module docstring).
        self.fused_corr = nn.Conv2d(num_disp * ch, num_disp * groups, 1,
                                    groups=1, bias=False)

    @property
    def out_channels(self) -> int:
        return self.num_disp * self.groups * 2

    def forward(self, left, right):
        d = self.num_disp
        # NOTE: cat([x] * d) rather than x.repeat(1, d, 1, 1) -- repeat
        # exports to a shape-dynamic Tile; cat is one static Concat.
        rl = self.reduce(left)                             # [B, G, h, w]
        rr = self.reduce(right)                            # [B, G, h, w]
        sub = (torch.cat([rl] * d, dim=1)                  # [B, D*G, h, w]
               - torch.cat(_shift_all(rr, d), dim=1))      # [B, D*G, h, w]
        big_left = torch.cat([left] * d, dim=1)            # [B, D*C, h, w]
        big_right = torch.cat(_shift_all(right, d), dim=1)  # [B, D*C, h, w]
        corr = self.fused_corr(big_left * big_right)       # [B, D*G, h, w]
        return torch.cat([sub, corr], dim=1)               # [B, 2*D*G, h, w]


class HailoStereoDense(HailoStereo):
    """A HailoStereo whose cost volume is the dense :class:`DenseCorrCostVolume`.

    Only ``cost="both"`` with ``refine="base"`` is supported -- the EdgeDepth-8
    configuration. Everything else is inherited unchanged from HailoStereo.
    """

    def __init__(self, num_disp=NUM_DISP, groups=8, match_ch=32, cost="both",
                 feat_ch=32, n_res=6, agg_ch=64, refine="base"):
        assert cost == "both", \
            f"HailoStereoDense only supports cost='both', got {cost!r}"
        assert refine == "base", \
            f"HailoStereoDense only supports refine='base', got {refine!r}"
        super().__init__(num_disp=num_disp, groups=groups, match_ch=match_ch,
                         cost=cost, feat_ch=feat_ch, n_res=n_res,
                         agg_ch=agg_ch, refine=refine)
        self.cost_volume = DenseCorrCostVolume(match_ch, num_disp, groups)


def convert_from_prered(prered_model: HailoStereoPrered) -> HailoStereoDense:
    """Build a HailoStereoDense that computes what `prered_model` computes.

    * ``cost_volume.reduce`` is copied directly (same shape).
    * ``cost_volume.fused_corr`` (grouped [D*G, C/G, 1, 1]) is expanded to
      the dense [D*G, D*C, 1, 1] block-diagonal weight.
    * ``aggregation.stem`` and every other parameter are copied unchanged
      (same output channel layout).
    """
    assert isinstance(prered_model, HailoStereoPrered), type(prered_model)
    assert isinstance(prered_model.cost_volume, PreReducedCostVolume), \
        type(prered_model.cost_volume)

    arch = dict(prered_model.arch)
    dense = HailoStereoDense(**arch)

    src = prered_model.state_dict()
    dst = dense.state_dict()

    fk = "cost_volume.fused_corr.weight"
    assert fk in src, "source state has no cost_volume.fused_corr.weight"
    gw = src[fk]  # [D*G, C/G, 1, 1], groups=D*G
    dw = expand_grouped_to_dense(gw)
    assert dw.shape == dst[fk].shape, \
        f"{tuple(dw.shape)} != {tuple(dst[fk].shape)}"
    dst[fk] = dw.clone()

    for key, value in src.items():
        if key == fk:
            continue
        assert key in dst, f"source key {key!r} has no dense counterpart"
        assert dst[key].shape == value.shape, \
            f"{key}: {tuple(value.shape)} != {tuple(dst[key].shape)}"
        dst[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    dense.load_state_dict(dst, strict=True)
    return dense


def convert_from_original(orig_model: HailoStereo) -> HailoStereoDense:
    """Build a HailoStereoDense from an original cost='both' model via the
    prered conversion, then the grouped->dense expansion (no training)."""
    return convert_from_prered(_prered_from_original(orig_model))
