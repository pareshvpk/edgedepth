"""
EdgeDepth-prered -- a mathematically identical copy of the EdgeDepth
matching path with the cost volume pre-reduced for Hailo-8 allocation.

Why this exists
--------------
EdgeDepth-grouped (src/model_grouped.py, GroupedCostVolume) fails Hailo-8
allocation: its cost volume is a 1536-channel tensor (cat[sub 768 | corr 768])
feeding one grouped 1x1 conv (1536 -> 384, groups=384), which the compiler
re-splits into 24 pieces and cannot route. This module computes the SAME
numbers with a narrower graph.

Math (authoritative spec)
-------------------------
Original per hypothesis d (see CostVolume in src/model.py, cost="both";
``reduce`` and ``reduce_corr`` are Conv2d(C, G, 1, groups=G, bias=False),
C=32, G=8, D=24)::

  sub_d  = reduce(left - shift_d(right))
  corr_d = reduce_corr(left * shift_d(right))

Because ``reduce`` is linear, has NO bias, is 1x1, and the shift pads with
zeros (reduce(0)=0)::

  reduce(left - shift_d(right)) == reduce(left) - shift_d(reduce(right))

New cost volume (class PreReducedCostVolume below)::

  rl = self.reduce(left)                   # [B, G, h, w]
  rr = self.reduce(right)                  # [B, G, h, w]
  sub  = torch.cat([rl]*D, 1) - torch.cat(_shift_all(rr, D), 1)  # [B, D*G]
  big_left  = torch.cat([left]*D, 1); big_right = torch.cat(_shift_all(right, D), 1)
  corr = self.fused_corr(big_left * big_right)  # Conv2d(D*C, D*G, 1, groups=D*G)

which returns ``torch.cat([sub, corr], 1)`` = ``[all sub (d-major) | all corr
(d-major)]`` -- EXACTLY the same output channel layout as GroupedCostVolume,
so the aggregation.stem input-channel permutation used by EdgeDepth-grouped
(_stem_channel_perm in src/model_grouped.py) applies unchanged.

Static-graph notes (Hailo parser needs a static graph)
------------------------------------------------------
* ``torch.cat([x] * D, 1)`` is used instead of ``x.repeat(1, D, 1, 1)``:
  repeat exports to a Tile op, which is shape-dynamic; cat of the same tensor
  D times is one static Concat (exactly as model_grouped.py does).
* ``_shift_all`` is reused by import, so the shift stays the pad-once /
  slice-many static form.

No retraining is involved: :func:`convert_from_original` only remaps weights.
"""

from __future__ import annotations

import copy
import sys
import pathlib

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, _shift_all, NUM_DISP  # noqa: E402
# Reuse the grouped conversion helpers so the channel layout contract cannot
# drift: the prered output layout is defined to equal the grouped layout.
from model_grouped import (  # noqa: E402
    _stem_channel_perm,
    load_original,
    strip_module_prefix,
)

__all__ = [
    "PreReducedCostVolume",
    "HailoStereoPrered",
    "convert_from_original",
    "_stem_channel_perm",
    "load_original",
    "strip_module_prefix",
]


class PreReducedCostVolume(nn.Module):
    """Narrow-layer equivalent of ``CostVolume(ch, num_disp, groups, cost="both")``.

    Only the ``"both"`` cost is supported -- that is the EdgeDepth arch, and
    the fused layout (sub half followed by corr half) is specific to it.

    The sub half is pre-reduced: ``reduce`` (a per-pixel linear map with no
    bias, so ``reduce(0)=0`` on the zero padding) is applied once per view and
    the shift happens in the narrow G-channel space. The corr half still needs
    the full C-channel product, so it uses one grouped 1x1 over the stacked
    products whose weight is ``reduce_corr.weight`` tiled D times.
    """

    def __init__(self, ch=32, num_disp=NUM_DISP, groups=8):
        super().__init__()
        assert ch % groups == 0, "channels must divide evenly into groups"
        self.ch = ch
        self.num_disp = num_disp
        self.groups = groups
        self.cost = "both"
        # Same weights as the original CostVolume.reduce (copied, not tiled).
        self.reduce = nn.Conv2d(ch, groups, 1, groups=groups, bias=False)
        # One grouped 1x1 over the D stacked products: each of the D*G groups
        # sees ch//groups input channels and emits one channel, so the weight
        # layout matches CostVolume.reduce_corr exactly, tiled D times.
        self.fused_corr = nn.Conv2d(num_disp * ch, num_disp * groups, 1,
                                    groups=num_disp * groups, bias=False)

    @property
    def out_channels(self) -> int:
        return self.num_disp * self.groups * 2

    def forward(self, left, right):
        d = self.num_disp
        # NOTE: cat([x] * d) rather than x.repeat(1, d, 1, 1) -- repeat
        # exports to a shape-dynamic Tile; cat is one static Concat.
        rl = self.reduce(left)                             # [B, G, h, w]
        rr = self.reduce(right)                            # [B, G, h, w]
        # Pre-reduced differences: one narrow ew_sub on D*G channels
        # (192 for the EdgeDepth arch) instead of a 1536-wide one.
        sub = (torch.cat([rl] * d, dim=1)                  # [B, D*G, h, w]
               - torch.cat(_shift_all(rr, d), dim=1))      # [B, D*G, h, w]
        big_left = torch.cat([left] * d, dim=1)            # [B, D*C, h, w]
        big_right = torch.cat(_shift_all(right, d), dim=1)  # [B, D*C, h, w]
        corr = self.fused_corr(big_left * big_right)       # [B, D*G, h, w]
        return torch.cat([sub, corr], dim=1)               # [B, 2*D*G, h, w]


class HailoStereoPrered(HailoStereo):
    """A HailoStereo whose cost volume is the pre-reduced :class:`PreReducedCostVolume`.

    Only ``cost="both"`` with ``refine="base"`` is supported -- the EdgeDepth
    configuration. Everything else (features, aggregation, soft-argmin,
    refinement ladder) is inherited unchanged from :class:`HailoStereo`.
    """

    def __init__(self, num_disp=NUM_DISP, groups=8, match_ch=32, cost="both",
                 feat_ch=32, n_res=6, agg_ch=64, refine="base"):
        assert cost == "both", \
            f"HailoStereoPrered only supports cost='both', got {cost!r}"
        assert refine == "base", \
            f"HailoStereoPrered only supports refine='base', got {refine!r}"
        super().__init__(num_disp=num_disp, groups=groups, match_ch=match_ch,
                         cost=cost, feat_ch=feat_ch, n_res=n_res,
                         agg_ch=agg_ch, refine=refine)
        # self.cost_volume is replaced by convert_from_original with a weight-
        # mapped PreReducedCostVolume; a fresh (random) one stands in until then.
        self.cost_volume = PreReducedCostVolume(match_ch, num_disp, groups)


def convert_from_original(orig_model: HailoStereo) -> HailoStereoPrered:
    """Build a :class:`HailoStereoPrered` that computes what `orig_model`
    computes, by remapping weights (no training).

    * ``cost_volume.reduce`` (``[G, C/G, 1, 1]`` grouped) is copied directly
      into ``cost_volume.reduce`` (same shape -- it now runs once per view
      instead of once per hypothesis).
    * ``cost_volume.reduce_corr`` is tiled D times into
      ``cost_volume.fused_corr``.
    * ``aggregation.stem``'s convolution input channels are permuted exactly
      as the grouped conversion does (same output layout: all-sub then
      all-corr, d-major), so no permute op is needed in the graph.
    * Every other parameter is copied unchanged.
    """
    assert isinstance(orig_model, HailoStereo), type(orig_model)
    assert orig_model.arch.get("cost") == "both", \
        f"need a cost='both' source, got {orig_model.arch.get('cost')!r}"
    assert orig_model.arch.get("refine", "base") == "base", \
        f"need a refine='base' source, got {orig_model.arch.get('refine')!r}"

    arch = dict(orig_model.arch)
    num_disp = arch["num_disp"]
    groups = arch["groups"]
    prered = HailoStereoPrered(**arch)

    src = orig_model.state_dict()
    dst = prered.state_dict()
    assert "cost_volume.reduce.weight" in src, \
        "source state has no cost_volume.reduce.weight"
    assert "cost_volume.reduce_corr.weight" in src, \
        "source state has no cost_volume.reduce_corr.weight (need cost='both')"

    reduce_w = src["cost_volume.reduce.weight"]        # [G, C/G, 1, 1]
    reduce_corr_w = src["cost_volume.reduce_corr.weight"]
    assert reduce_w.shape[0] == groups and reduce_corr_w.shape[0] == groups

    assert dst["cost_volume.reduce.weight"].shape == reduce_w.shape, \
        f"{tuple(reduce_w.shape)} != " \
        f"{tuple(dst['cost_volume.reduce.weight'].shape)}"
    dst["cost_volume.reduce.weight"] = reduce_w.clone()

    fused_corr_w = torch.cat([reduce_corr_w] * num_disp, dim=0)
    assert fused_corr_w.shape == dst["cost_volume.fused_corr.weight"].shape, \
        f"{tuple(fused_corr_w.shape)} != " \
        f"{tuple(dst['cost_volume.fused_corr.weight'].shape)}"
    dst["cost_volume.fused_corr.weight"] = fused_corr_w.clone()

    # Reorder the stem to consume prered [all-sub | all-corr] order as if it
    # were the original per-d [sub | corr] order -- identical to the grouped
    # conversion because the output channel layout is identical.
    # conv_bn_act builds the stem as Sequential(Conv2d, BatchNorm2d,
    # LeakyReLU), so the convolution is aggregation.stem[0] and only its
    # INPUT channels move; BatchNorm stats (output channels) are untouched.
    stem_key = "aggregation.stem.0.weight"             # [agg_ch, 2*D*G, 3, 3]
    assert stem_key in src, "source state has no aggregation.stem.0.weight"
    perm = _stem_channel_perm(num_disp, groups)
    stem_w = src[stem_key]
    assert stem_w.shape[1] == 2 * num_disp * groups, \
        f"stem cin {stem_w.shape[1]} != 2*D*G={2 * num_disp * groups}"
    reordered = torch.empty_like(stem_w)
    reordered[:, perm, :, :] = stem_w
    dst[stem_key] = reordered

    for key, value in src.items():
        if key in ("cost_volume.reduce.weight",
                   "cost_volume.reduce_corr.weight", stem_key):
            continue
        assert key in dst, f"source key {key!r} has no prered counterpart"
        assert dst[key].shape == value.shape, \
            f"{key}: {tuple(value.shape)} != {tuple(dst[key].shape)}"
        dst[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    prered.load_state_dict(dst, strict=True)
    return prered
