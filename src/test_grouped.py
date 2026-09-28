"""
Equivalence tests for the grouped cost volume (EdgeDepth-8-grouped).

GroupedCostVolume + the stem input-channel permutation must compute exactly
what CostVolume(cost="both") + the original stem computes, for any weights --
no retraining is involved, so this is checked on randomly initialised models.

Run directly:   python src/test_grouped.py
Or via pytest:  pytest src/test_grouped.py -v
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, NUM_DISP  # noqa: E402
from model_grouped import (  # noqa: E402
    GroupedCostVolume,
    HailoStereoGrouped,
    convert_from_original,
    _stem_channel_perm,
)


def _random_both(seed=0, **kw) -> HailoStereo:
    torch.manual_seed(seed)
    kw.setdefault("cost", "both")
    return HailoStereo(**kw).eval()


def test_grouped_cost_volume_matches_interleaved():
    """Fused conv output, reordered to the original per-d layout, must equal
    the original cost volume bit-for-bit (same subtractions, same products,
    same grouped weights -- only the loop is gone)."""
    torch.manual_seed(0)
    orig = _random_both()
    grouped = convert_from_original(orig).eval()
    d, g = orig.num_disp, orig.cost_volume.groups
    idx = []
    for dd in range(d):
        for k in range(2):
            for gg in range(g):
                idx.append(k * d * g + dd * g + gg)

    torch.manual_seed(1)
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)
    with torch.no_grad():
        a = orig.cost_volume(left, right)
        b = grouped.cost_volume(left, right)[:, idx]
    assert a.shape == b.shape == (1, d * g * 2, 8, 16)
    assert torch.equal(a, b), \
        f"grouped cost volume differs: maxdiff {float((a - b).abs().max()):.3e}"
    print("  ok  grouped cost volume matches original bit-for-bit")


def test_converted_model_matches_random_input():
    """End-to-end disparity on a random small input (1x3x64x128, divisible by
    16 as HailoStereo requires). The only allowed slack is float summation
    order in the channel-permuted stem convolution."""
    orig = _random_both(seed=0)
    grouped = convert_from_original(orig).eval()
    torch.manual_seed(7)
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        a = orig(left, right)
        b = grouped(left, right)
    assert a.shape == b.shape == (1, 1, 64, 128), (a.shape, b.shape)
    diff = float((a - b).abs().max())
    assert diff < 1e-4, f"end-to-end maxdiff {diff:.3e} px too large"
    print(f"  ok  converted model matches: maxdiff {diff:.3e} px")


def test_output_shape_and_channel_counts():
    """Shape contract: cost volume [B, 2*D*G, h, w], disparity [B, 1, H, W]."""
    torch.manual_seed(0)
    m = convert_from_original(_random_both()).eval()
    assert isinstance(m, HailoStereoGrouped)
    assert isinstance(m.cost_volume, GroupedCostVolume)
    assert m.cost_volume.out_channels == 2 * NUM_DISP * 8 == 384
    assert m.cost_volume.fused.weight.shape == (384, 4, 1, 1)
    assert m.cost_volume.fused.groups == 384
    assert m.aggregation.stem[0].weight.shape[1] == 384
    torch.manual_seed(2)
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128), out.shape
    assert torch.isfinite(out).all()
    print("  ok  shapes: cost [1,384,8,16], disparity [1,1,64,128]")


def test_stem_perm_is_consistent():
    """The stem permutation must be a permutation, and applying it twice
    (grouped order -> original order -> grouped order) must round-trip."""
    n = 2 * NUM_DISP * 8
    perm = _stem_channel_perm(NUM_DISP, 8)
    assert sorted(perm) == list(range(n)), "not a permutation"
    x = torch.randn(1, n, 4, 4)
    reordered = torch.empty_like(x)
    reordered[:, perm, :, :] = x
    back = torch.empty_like(x)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    back[:, inv, :, :] = reordered
    assert torch.equal(back, x)
    print("  ok  stem channel permutation round-trips")


def test_rejects_non_both_cost():
    """Only cost='both' (the EdgeDepth-8 arch) can be grouped."""
    torch.manual_seed(0)
    try:
        HailoStereoGrouped(cost="sub")
    except AssertionError:
        print("  ok  HailoStereoGrouped(cost='sub') refused")
    else:
        raise AssertionError("HailoStereoGrouped(cost='sub') should assert")
    try:
        convert_from_original(_random_both(cost="sub"))
    except AssertionError:
        print("  ok  convert_from_original(cost='sub') refused")
    else:
        raise AssertionError("convert_from_original(cost='sub') should assert")


if __name__ == "__main__":
    for fn in (test_grouped_cost_volume_matches_interleaved,
               test_converted_model_matches_random_input,
               test_output_shape_and_channel_counts,
               test_stem_perm_is_consistent,
               test_rejects_non_both_cost):
        fn()
    print("grouped: all tests passed")
