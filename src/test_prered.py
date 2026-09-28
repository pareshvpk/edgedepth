"""
Equivalence tests for the pre-reduced cost volume (EdgeDepth-8-prered).

PreReducedCostVolume + the stem input-channel permutation must compute what
CostVolume(cost="both") + the original stem computes, for any weights -- no
retraining is involved, so this is checked on randomly initialised models.

The sub half is mathematically identical but reorders floating-point summation
(reduce(left - shift(right)) vs reduce(left) - shift(reduce(right))), so the
cost-volume comparison allows a tight tolerance instead of bit equality. The
corr half is the same products through the same grouped weights.

Run directly:   python src/test_prered.py
Or via pytest:  pytest src/test_prered.py -v
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, NUM_DISP  # noqa: E402
from model_prered import (  # noqa: E402
    PreReducedCostVolume,
    HailoStereoPrered,
    convert_from_original,
    _stem_channel_perm,
)


def _random_both(seed=0, **kw) -> HailoStereo:
    torch.manual_seed(seed)
    kw.setdefault("cost", "both")
    return HailoStereo(**kw).eval()


def test_prered_cost_volume_matches_interleaved():
    """Pre-reduced conv output, reordered to the original per-d layout, must
    equal the original cost volume (same subtractions/products up to float
    summation order in the sub half, same grouped weights throughout)."""
    torch.manual_seed(0)
    orig = _random_both()
    prered = convert_from_original(orig).eval()
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
        b = prered.cost_volume(left, right)[:, idx]
    assert a.shape == b.shape == (1, d * g * 2, 8, 16)
    diff = float((a - b).abs().max())
    assert diff < 1e-5, \
        f"prered cost volume differs: maxdiff {diff:.3e}"
    print(f"  ok  prered cost volume matches original: maxdiff {diff:.3e}")


def test_converted_model_matches_random_input():
    """End-to-end disparity on a random small input (1x3x64x128, divisible by
    16 as HailoStereo requires). The allowed slack covers float summation
    order in the pre-reduced sub path plus the channel-permuted stem."""
    orig = _random_both(seed=0)
    prered = convert_from_original(orig).eval()
    torch.manual_seed(7)
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        a = orig(left, right)
        b = prered(left, right)
    assert a.shape == b.shape == (1, 1, 64, 128), (a.shape, b.shape)
    diff = float((a - b).abs().max())
    assert diff < 1e-4, f"end-to-end maxdiff {diff:.3e} px too large"
    print(f"  ok  converted model matches: maxdiff {diff:.3e} px")


def test_output_shape_and_channel_counts():
    """Shape contract: cost volume [B, 2*D*G, h, w], disparity [B, 1, H, W];
    narrow widths: reduce [G, C/G, 1, 1], fused_corr [D*G, C/G, 1, 1]."""
    torch.manual_seed(0)
    m = convert_from_original(_random_both()).eval()
    assert isinstance(m, HailoStereoPrered)
    assert isinstance(m.cost_volume, PreReducedCostVolume)
    assert m.cost_volume.out_channels == 2 * NUM_DISP * 8 == 384
    assert m.cost_volume.reduce.weight.shape == (8, 4, 1, 1)
    assert m.cost_volume.fused_corr.weight.shape == (192, 4, 1, 1)
    assert m.cost_volume.fused_corr.groups == 192
    assert m.aggregation.stem[0].weight.shape[1] == 384
    torch.manual_seed(2)
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128), out.shape
    assert torch.isfinite(out).all()
    print("  ok  shapes: cost [1,384,8,16], disparity [1,1,64,128]")


def test_stem_perm_is_consistent():
    """The stem permutation must be a permutation, and applying it twice
    (prered order -> original order -> prered order) must round-trip."""
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
    """Only cost='both' (the EdgeDepth-8 arch) can be pre-reduced."""
    torch.manual_seed(0)
    try:
        HailoStereoPrered(cost="sub")
    except AssertionError:
        print("  ok  HailoStereoPrered(cost='sub') refused")
    else:
        raise AssertionError("HailoStereoPrered(cost='sub') should assert")
    try:
        convert_from_original(_random_both(cost="sub"))
    except AssertionError:
        print("  ok  convert_from_original(cost='sub') refused")
    else:
        raise AssertionError("convert_from_original(cost='sub') should assert")


if __name__ == "__main__":
    for fn in (test_prered_cost_volume_matches_interleaved,
               test_converted_model_matches_random_input,
               test_output_shape_and_channel_counts,
               test_stem_perm_is_consistent,
               test_rejects_non_both_cost):
        fn()
    print("prered: all tests passed")
