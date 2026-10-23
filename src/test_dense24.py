"""
Equivalence tests for the single-shift dense cost volume (EdgeDepth).

The dense24 fused_sub must compute what DenseCorrCostVolume's reduce-then-shift
sub path computes, for any weights -- the expansion is exact (block-diagonal,
zeros elsewhere). The end-to-end model comparison inherits float summation
order, hence a slightly looser tolerance.

Run directly:   python src/test_dense24.py
Or via pytest:  pytest src/test_dense24.py -v
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, NUM_DISP  # noqa: E402
from model_dense import convert_from_original as dense_from_original  # noqa: E402
from model_dense24 import (  # noqa: E402
    DenseSubCorrCostVolume,
    HailoStereoDense24,
    convert_from_dense,
    convert_from_original,
    _stem_channel_perm,
)


def _random_both(seed=0, **kw) -> HailoStereo:
    torch.manual_seed(seed)
    kw.setdefault("cost", "both")
    return HailoStereo(**kw).eval()


def test_cost_volume_matches_dense():
    """Cost volume dense24 == dense on random input (<1e-4)."""
    from model_dense import DenseCorrCostVolume
    torch.manual_seed(0)
    d, g, ch = NUM_DISP, 8, 32
    dense_cv = DenseCorrCostVolume(ch, d, g).eval()
    m24 = DenseSubCorrCostVolume(ch, d, g).eval()
    with torch.no_grad():
        m24.fused_corr.weight.copy_(dense_cv.fused_corr.weight)
        from model_dense import expand_grouped_to_dense
        tiled = torch.cat([dense_cv.reduce.weight] * d, dim=0)
        m24.fused_sub.weight.copy_(expand_grouped_to_dense(tiled))
    # Sparsity contract: each fused_sub row has exactly k=C//G nonzeros at
    # cols d*C+g*4 .. d*C+g*4+3.
    k = ch // g
    w = m24.fused_sub.weight[..., 0, 0]
    assert w.shape == (d * g, d * ch), w.shape
    nnz = (w != 0).sum(dim=1)
    assert bool((nnz == k).all()), f"rows must have {k} nonzeros: {nnz.unique()}"
    for dd in range(d):
        for gg in range(g):
            row = w[dd * g + gg]
            assert bool((row[dd * ch + gg * k:dd * ch + (gg + 1) * k] != 0).all())
            before = row[:dd * ch + gg * k]
            after = row[dd * ch + (gg + 1) * k:]
            assert bool((before == 0).all() and (after == 0).all())
    torch.manual_seed(1)
    left = torch.randn(2, ch, 6, 8)
    right = torch.randn(2, ch, 6, 8)
    with torch.no_grad():
        a = dense_cv(left, right)
        b = m24(left, right)
    assert a.shape == b.shape == (2, 2 * d * g, 6, 8)
    diff = float((a - b).abs().max())
    assert diff < 1e-4, f"dense24 cost volume differs: maxdiff {diff:.3e}"
    print(f"  ok  dense24 cost volume == dense: maxdiff {diff:.3e}")


def test_converted_model_matches_dense_and_original():
    """End-to-end disparity: dense24 == dense (tight) and dense24 == original."""
    orig = _random_both(seed=0)
    dense = dense_from_original(orig).eval()
    m24 = convert_from_dense(dense).eval()
    m24b = convert_from_original(orig).eval()
    torch.manual_seed(7)
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        a = orig(left, right)
        p = dense(left, right)
        b = m24(left, right)
        c = m24b(left, right)
    assert b.shape == (1, 1, 64, 128), b.shape
    dp = float((p - b).abs().max())
    do = float((a - b).abs().max())
    dc = float((b - c).abs().max())
    assert dp < 1e-4, f"dense24 vs dense maxdiff {dp:.3e} px too large"
    assert do < 1e-4, f"dense24 vs original maxdiff {do:.3e} px too large"
    assert dc == 0.0, f"two dense24 paths differ: {dc:.3e}"
    print(f"  ok  dense24==dense {dp:.3e} px, dense24==orig {do:.3e} px")


def test_output_shape_and_channel_counts():
    """Shape contract: cost volume [B, 2*D*G, h, w], disparity [B, 1, H, W];
    fused_sub/fused_corr [D*G, D*C, 1, 1] groups=1; no reduce remains."""
    torch.manual_seed(0)
    m = convert_from_original(_random_both()).eval()
    assert isinstance(m, HailoStereoDense24)
    assert isinstance(m.cost_volume, DenseSubCorrCostVolume)
    assert m.cost_volume.out_channels == 2 * NUM_DISP * 8 == 384
    assert not hasattr(m.cost_volume, "reduce")
    assert m.cost_volume.fused_sub.weight.shape == (192, 768, 1, 1)
    assert m.cost_volume.fused_sub.groups == 1
    assert m.cost_volume.fused_corr.weight.shape == (192, 768, 1, 1)
    assert m.cost_volume.fused_corr.groups == 1
    assert m.aggregation.stem[0].weight.shape[1] == 384
    # Dense stem must be reused unchanged (same layout).
    dense = dense_from_original(_random_both(seed=0)).eval()
    torch.manual_seed(0)
    m2 = convert_from_dense(dense).eval()
    assert torch.equal(m2.aggregation.stem[0].weight,
                       dense.aggregation.stem[0].weight)
    assert torch.equal(m2.cost_volume.fused_corr.weight,
                       dense.cost_volume.fused_corr.weight)
    torch.manual_seed(2)
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128), out.shape
    assert torch.isfinite(out).all()
    print("  ok  shapes: fused_sub/corr [192,768,1,1] groups=1, disparity [1,1,64,128]")


def test_stem_perm_is_consistent():
    """The reused stem permutation must be a permutation and round-trip."""
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
    """Only cost='both' (the EdgeDepth arch) can be converted."""
    torch.manual_seed(0)
    try:
        HailoStereoDense24(cost="sub")
    except AssertionError:
        print("  ok  HailoStereoDense24(cost='sub') refused")
    else:
        raise AssertionError("HailoStereoDense24(cost='sub') should assert")
    try:
        convert_from_original(_random_both(cost="sub"))
    except AssertionError:
        print("  ok  convert_from_original(cost='sub') refused")
    else:
        raise AssertionError("convert_from_original(cost='sub') should assert")


if __name__ == "__main__":
    for fn in (test_cost_volume_matches_dense,
               test_converted_model_matches_dense_and_original,
               test_output_shape_and_channel_counts,
               test_stem_perm_is_consistent,
               test_rejects_non_both_cost):
        fn()
    print("dense24: all tests passed")
