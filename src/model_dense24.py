"""
EdgeDepth -- exact single-shift variant of EdgeDepth-dense.

Why this exists
--------------
The Hailo-8 compiler cannot split EdgeDepth-dense into contexts. The cause
is 48 Slice layers: ``_shift_all`` is called twice, once on ``reduce(right)``
(sub path) and once on ``right`` (corr path). This module computes the SAME
numbers with only ONE set of 24 shift slices.

Math (D=24, C=32, G=8, C//G=4)
------------------------------
``reduce`` = Conv2d(C, G, 1, groups=G, bias=False) has no bias and the shift
pads with zeros, so shift_d(reduce(R)) == reduce(shift_d(R)), and reduce is
linear: reduce(L) - reduce(R_d) == reduce(L - R_d).

New forward (``DenseSubCorrCostVolume``)::

  big_left  = cat([left] * D, 1)              # [B, D*C, h, w]
  big_right = cat(_shift_all(right, D), 1)     # [B, D*C, h, w]  the ONLY 24 slices
  sub  = fused_sub(big_left - big_right)      # Conv2d(D*C, D*G, 1, bias=False) DENSE
  corr = fused_corr(big_left * big_right)     # unchanged dense 768->192
  return cat([sub, corr], 1)                  # same layout as model_dense

``fused_sub.weight[d*G + g, d*C + g*4 + k] = reduce.weight[g, k, 0, 0]``
for k in 0..3, all other entries 0 (the same block-diagonal pattern as the
reduce conv repeated per d; built with ``expand_grouped_to_dense`` over the
D-tiled reduce weight). No ``reduce`` call remains in forward. Output channel
order equals model_dense's ``cat([sub, corr])`` exactly, so the aggregation
stem is untouched.

No retraining: convert_from_dense / convert_from_original only remap weights.
"""

from __future__ import annotations

import copy
import sys
import pathlib

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, _shift_all, NUM_DISP  # noqa: E402
# Reuse the dense helpers so the channel-layout contract cannot drift.
from model_dense import (  # noqa: E402
    HailoStereoDense,
    DenseCorrCostVolume,
    convert_from_original as _dense_from_original,
    expand_grouped_to_dense,
)
from model_grouped import (  # noqa: E402
    _stem_channel_perm,
    load_original,
    strip_module_prefix,
)

__all__ = [
    "DenseSubCorrCostVolume",
    "HailoStereoDense24",
    "convert_from_dense",
    "convert_from_original",
    "_stem_channel_perm",
    "load_original",
    "strip_module_prefix",
    "expand_grouped_to_dense",
]


class DenseSubCorrCostVolume(nn.Module):
    """Drop-in copy of DenseCorrCostVolume with a single 24-slice shift.

    Only the ``"both"`` cost is supported -- the EdgeDepth arch. Output
    layout is ``torch.cat([sub, corr], 1)`` = ``[all sub (d-major) | all
    corr (d-major)]``, exactly like DenseCorrCostVolume, so the same
    stem input-channel permutation applies (and the stem weights are
    copied unchanged from the dense model).
    """

    def __init__(self, ch=32, num_disp=NUM_DISP, groups=8):
        super().__init__()
        assert ch % groups == 0, "channels must divide evenly into groups"
        self.ch = ch
        self.num_disp = num_disp
        self.groups = groups
        self.cost = "both"
        # Dense twin of DenseCorrCostVolume.reduce, repeated per hypothesis:
        # block-diagonal, row d*G+g nonzero only at cols d*C+g*4 .. d*C+g*4+3.
        self.fused_sub = nn.Conv2d(num_disp * ch, num_disp * groups, 1,
                                   groups=1, bias=False)
        # Same dense corr 1x1 as DenseCorrCostVolume.fused_corr.
        self.fused_corr = nn.Conv2d(num_disp * ch, num_disp * groups, 1,
                                    groups=1, bias=False)

    @property
    def out_channels(self) -> int:
        return self.num_disp * self.groups * 2

    def forward(self, left, right):
        d = self.num_disp
        # NOTE: cat([x] * d) rather than x.repeat(1, d, 1, 1) -- repeat
        # exports to a shape-dynamic Tile; cat is one static Concat.
        big_left = torch.cat([left] * d, dim=1)            # [B, D*C, h, w]
        big_right = torch.cat(_shift_all(right, d), dim=1)  # [B, D*C, h, w]
        sub = self.fused_sub(big_left - big_right)         # [B, D*G, h, w]
        corr = self.fused_corr(big_left * big_right)       # [B, D*G, h, w]
        return torch.cat([sub, corr], dim=1)               # [B, 2*D*G, h, w]


class HailoStereoDense24(HailoStereo):
    """A HailoStereo whose cost volume is the single-shift
    :class:`DenseSubCorrCostVolume`.

    Only ``cost="both"`` with ``refine="base"`` is supported -- the EdgeDepth
    configuration. Everything else is inherited unchanged from HailoStereo.
    """

    def __init__(self, num_disp=NUM_DISP, groups=8, match_ch=32, cost="both",
                 feat_ch=32, n_res=6, agg_ch=64, refine="base"):
        assert cost == "both", \
            f"HailoStereoDense24 only supports cost='both', got {cost!r}"
        assert refine == "base", \
            f"HailoStereoDense24 only supports refine='base', got {refine!r}"
        super().__init__(num_disp=num_disp, groups=groups, match_ch=match_ch,
                         cost=cost, feat_ch=feat_ch, n_res=n_res,
                         agg_ch=agg_ch, refine=refine)
        self.cost_volume = DenseSubCorrCostVolume(match_ch, num_disp, groups)


def convert_from_dense(dense_model: HailoStereoDense) -> HailoStereoDense24:
    """Build a HailoStereoDense24 that computes what `dense_model` computes.

    * ``cost_volume.fused_corr`` (dense [D*G, D*C, 1, 1]) is copied directly
      (same shape -- the corr path is unchanged).
    * ``cost_volume.fused_sub`` (dense [D*G, D*C, 1, 1]) is the
      block-diagonal expansion of ``cost_volume.reduce`` ([G, C/G, 1, 1])
      tiled D times: ``expand_grouped_to_dense(cat([reduce] * D))`` gives
      row d*G+g nonzero exactly at cols d*C+g*4 .. d*C+g*4+3.
    * ``aggregation.stem`` and every other parameter are copied unchanged
      (same output channel layout).
    """
    assert isinstance(dense_model, HailoStereoDense), type(dense_model)
    assert isinstance(dense_model.cost_volume, DenseCorrCostVolume), \
        type(dense_model.cost_volume)

    arch = dict(dense_model.arch)
    model = HailoStereoDense24(**arch)
    num_disp = arch["num_disp"]

    src = dense_model.state_dict()
    dst = model.state_dict()

    fk = "cost_volume.fused_corr.weight"
    assert fk in src, "source state has no cost_volume.fused_corr.weight"
    assert dst[fk].shape == src[fk].shape, \
        f"{tuple(src[fk].shape)} != {tuple(dst[fk].shape)}"
    dst[fk] = src[fk].clone()

    rk = "cost_volume.reduce.weight"
    assert rk in src, "source state has no cost_volume.reduce.weight"
    rw = src[rk]  # [G, C/G, 1, 1], groups=G
    tiled = torch.cat([rw] * num_disp, dim=0)  # [D*G, C/G, 1, 1]
    sw = expand_grouped_to_dense(tiled)
    sk = "cost_volume.fused_sub.weight"
    assert sw.shape == dst[sk].shape, \
        f"{tuple(sw.shape)} != {tuple(dst[sk].shape)}"
    dst[sk] = sw.clone()

    for key, value in src.items():
        if key in (fk, rk):
            continue
        assert key in dst, f"source key {key!r} has no dense24 counterpart"
        assert dst[key].shape == value.shape, \
            f"{key}: {tuple(value.shape)} != {tuple(dst[key].shape)}"
        dst[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    model.load_state_dict(dst, strict=True)
    return model


def convert_from_original(orig_model: HailoStereo) -> HailoStereoDense24:
    """Build a HailoStereoDense24 from an original cost='both' model via the
    dense conversion, then the reduce->fused_sub expansion (no training)."""
    return convert_from_dense(_dense_from_original(orig_model))
