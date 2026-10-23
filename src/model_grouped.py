"""
EdgeDepth-grouped -- a mathematically identical copy of the EdgeDepth
matching path with the 24-branch cost volume fused into wide grouped layers.

Why this exists
--------------
CostVolume.forward loops over 24 disparity hypotheses. Per hypothesis it runs
an ew_sub -> grouped 1x1 (``reduce``) and an ew_mult -> grouped 1x1
(``reduce_corr``), then concatenates 48 small tensors. That is ~100 tiny
layers in the Hailo graph, and the 'both' arch failed Hailo-8 allocation.

This module computes the SAME numbers as a handful of wide layers:

1. ``L = torch.cat([left] * D, 1)``           -> [B, D*C, h, w]
2. ``R = torch.cat(_shift_all(right, D), 1)``  -> [B, D*C, h, w]
3. ``S = L - R`` (one ew_sub), ``M = L * R`` (one ew_mult)
4. ``X = torch.cat([S, M], 1)`` -> one
   ``nn.Conv2d(2*D*C, 2*D*G, 1, groups=2*D*G, bias=False)``
   whose first D*G groups copy ``reduce.weight`` tiled D times and whose
   next D*G groups copy ``reduce_corr.weight`` tiled D times.

The fused convolution emits ``[all sub (d-major, G each) | all corr]`` while
the original emits per-d ``[sub G | corr G]``. No interleave/permutation op is
added to the graph; instead the INPUT channels of ``aggregation.stem``'s
convolution are permuted at conversion time, so the downstream result is
identical.

Static-graph notes (Hailo parser needs a static graph)
------------------------------------------------------
* ``torch.cat([left] * D, 1)`` is used instead of ``left.repeat(1, D, 1, 1)``:
  repeat exports to a Tile op, which is shape-dynamic; cat of the same tensor
  D times is one static Concat.
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


class GroupedCostVolume(nn.Module):
    """Wide-layer equivalent of ``CostVolume(ch, num_disp, groups, cost="both")``.

    Only the ``"both"`` cost is supported -- that is the EdgeDepth arch, and
    the fused layout (sub half followed by corr half) is specific to it.
    """

    def __init__(self, ch=32, num_disp=NUM_DISP, groups=8):
        super().__init__()
        assert ch % groups == 0, "channels must divide evenly into groups"
        self.ch = ch
        self.num_disp = num_disp
        self.groups = groups
        self.cost = "both"
        # One grouped 1x1 over cat([S, M]): the first D*G groups reduce the
        # stacked differences, the next D*G groups reduce the stacked products.
        # Each group sees ch//groups input channels and emits one channel, so
        # the weight layout matches CostVolume.reduce / reduce_corr exactly.
        self.fused = nn.Conv2d(2 * num_disp * ch, 2 * num_disp * groups, 1,
                               groups=2 * num_disp * groups, bias=False)

    @property
    def out_channels(self) -> int:
        return self.num_disp * self.groups * 2

    def forward(self, left, right):
        d = self.num_disp
        # NOTE: cat([left] * d) rather than left.repeat(1, d, 1, 1) -- repeat
        # exports to a shape-dynamic Tile; cat is one static Concat.
        big_left = torch.cat([left] * d, dim=1)          # [B, D*C, h, w]
        big_right = torch.cat(_shift_all(right, d), dim=1)  # [B, D*C, h, w]
        sub = big_left - big_right                        # one ew_sub
        corr = big_left * big_right                        # one ew_mult
        return self.fused(torch.cat([sub, corr], dim=1))  # [B, 2*D*G, h, w]


class HailoStereoGrouped(HailoStereo):
    """A HailoStereo whose cost volume is the fused :class:`GroupedCostVolume`.

    Only ``cost="both"`` with ``refine="base"`` is supported -- the EdgeDepth
    configuration. Everything else (features, aggregation, soft-argmin,
    refinement ladder) is inherited unchanged from :class:`HailoStereo`.
    """

    def __init__(self, num_disp=NUM_DISP, groups=8, match_ch=32, cost="both",
                 feat_ch=32, n_res=6, agg_ch=64, refine="base"):
        assert cost == "both", \
            f"HailoStereoGrouped only supports cost='both', got {cost!r}"
        assert refine == "base", \
            f"HailoStereoGrouped only supports refine='base', got {refine!r}"
        super().__init__(num_disp=num_disp, groups=groups, match_ch=match_ch,
                         cost=cost, feat_ch=feat_ch, n_res=n_res,
                         agg_ch=agg_ch, refine=refine)
        # self.cost_volume is replaced by convert_from_original with a weight-
        # mapped GroupedCostVolume; a fresh (random) one stands in until then.
        self.cost_volume = GroupedCostVolume(match_ch, num_disp, groups)


def _stem_channel_perm(num_disp: int, groups: int) -> list[int]:
    """For each ORIGINAL cost-volume output channel, the GROUPED channel that
    carries the same value.

    Original order is per-d interleaved: channel ``d*2*G + k*G + g`` holds
    disparity d, kind k (0 = sub, 1 = corr), group g. Grouped order is all-sub
    then all-corr: ``k*D*G + d*G + g``. The aggregation stem's input channels
    are permuted by this map at conversion time so no permute op is needed in
    the graph.
    """
    perm = []
    for d in range(num_disp):
        for k in range(2):
            for g in range(groups):
                perm.append(k * num_disp * groups + d * groups + g)
    return perm


def convert_from_original(orig_model: HailoStereo) -> HailoStereoGrouped:
    """Build a :class:`HailoStereoGrouped` that computes what `orig_model`
    computes, by remapping weights (no training).

    * ``cost_volume.reduce`` (``[G, C/G, 1, 1]`` grouped) is tiled D times
      into the first D*G groups of ``cost_volume.fused``; ``reduce_corr``
      is tiled D times into the next D*G groups.
    * ``aggregation.stem``'s convolution input channels are permuted so the
      stem sees the original per-d ``[sub | corr]`` order.
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
    grouped = HailoStereoGrouped(**arch)

    src = orig_model.state_dict()
    dst = grouped.state_dict()
    assert "cost_volume.reduce.weight" in src, \
        "source state has no cost_volume.reduce.weight"
    assert "cost_volume.reduce_corr.weight" in src, \
        "source state has no cost_volume.reduce_corr.weight (need cost='both')"

    reduce_w = src["cost_volume.reduce.weight"]        # [G, C/G, 1, 1]
    reduce_corr_w = src["cost_volume.reduce_corr.weight"]
    assert reduce_w.shape[0] == groups and reduce_corr_w.shape[0] == groups

    fused_w = torch.cat([reduce_w] * num_disp + [reduce_corr_w] * num_disp,
                        dim=0)
    assert fused_w.shape == dst["cost_volume.fused.weight"].shape, \
        f"{tuple(fused_w.shape)} != {tuple(dst['cost_volume.fused.weight'].shape)}"
    dst["cost_volume.fused.weight"] = fused_w.clone()

    # Reorder the stem to consume grouped [all-sub | all-corr] order as if it
    # were the original per-d [sub | corr] order. conv_bn_act builds the stem
    # as Sequential(Conv2d, BatchNorm2d, LeakyReLU), so the convolution is
    # aggregation.stem[0] and only its INPUT channels move; BatchNorm stats
    # (output channels) are untouched.
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
        assert key in dst, f"source key {key!r} has no grouped counterpart"
        assert dst[key].shape == value.shape, \
            f"{key}: {tuple(value.shape)} != {tuple(dst[key].shape)}"
        dst[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)

    grouped.load_state_dict(dst, strict=True)
    return grouped


def strip_module_prefix(state: dict) -> dict:
    """Drop a leading ``module.`` from every key that has one (DataParallel
    checkpoints), so strict loads work the same as in eval/export."""
    if any(k.startswith("module.") for k in state):
        return {(k[7:] if k.startswith("module.") else k): v
                for k, v in state.items()}
    return state


def load_original(ckpt_path: str) -> HailoStereo:
    """Load a source checkpoint the way src/eval_kitti.py does (``blob.get(
    "model", blob)``), tolerating a ``module.`` prefix, and return an eval
    model. Raises if the arch is not the supported EdgeDepth one."""
    from model import build_model  # noqa: E402  (local import: keep module import light)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = blob.get("model", blob) if isinstance(blob, dict) else blob
    raw = strip_module_prefix(raw)
    model = build_model(blob if isinstance(blob, dict) else None)
    if isinstance(blob, dict) and "arch" not in blob:
        # pre-arch-config checkpoints default to cost="sub"; those are out of
        # scope for the grouped conversion.
        raise AssertionError("checkpoint carries no arch entry; "
                             "need an explicit cost='both' arch")
    model.load_state_dict(raw, strict=True)
    assert model.arch.get("cost") == "both", \
        f"need a cost='both' source, got {model.arch.get('cost')!r}"
    assert model.arch.get("refine", "base") == "base", \
        f"need a refine='base' source, got {model.arch.get('refine')!r}"
    return model.eval()
