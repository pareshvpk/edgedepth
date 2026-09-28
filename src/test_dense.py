"""
Equivalence tests for the dense-1x1 cost volume (EdgeDepth-8-dense).

The dense fused_corr must compute what the grouped fused_corr of
PreReducedCostVolume computes, for any weights -- the expansion is exact
(block-diagonal, zeros elsewhere), so the conv comparison is tight. The
end-to-end model comparison inherits the float summation order of the
pre-reduced sub path, hence a slightly looser tolerance.

Run directly:   python src/test_dense.py
Or via pytest:  pytest src/test_dense.py -v
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import HailoStereo, NUM_DISP  # noqa: E402
from model_prered import convert_from_original as prered_from_original  # noqa: E402
from model_dense import (  # noqa: E402
    DenseCorrCostVolume,
    HailoStereoDense,
    convert_from_prered,
    convert_from_original,
    expand_grouped_to_dense,
    _stem_channel_perm,
)


def _random_both(seed=0, **kw) -> HailoStereo:
    torch.manual_seed(seed)
    kw.setdefault("cost", "both")
    return HailoStereo(**kw).eval()


def test_dense_conv_equals_grouped_conv():
    """Dense expansion == grouped conv on random input (<1e-5)."""
    torch.manual_seed(0)
    d, g, ch = NUM_DISP, 8, 32
    grouped = torch.nn.Conv2d(d * ch, d * g, 1, groups=d * g, bias=False)
    dense_w = expand_grouped_to_dense(grouped.weight.detach())
    dense = torch.nn.Conv2d(d * ch, d * g, 1, groups=1, bias=False)
    dense.weight.data.copy_(dense_w)
    # Sparsity contract: each output row has exactly k=C//G nonzeros.
    k = ch // g
    nnz = (dense_w[..., 0, 0] != 0).sum(dim=1)
    assert dense_w.shape == (d * g, d * ch, 1, 1), dense_w.shape
    assert bool((nnz == k).all()), f"rows must have {k} nonzeros: {nnz.unique()}"
    torch.manual_seed(1)
    x = torch.randn(2, d * ch, 6, 8)
    with torch.no_grad():
        a = grouped(x)
        b = dense(x)
    assert a.shape == b.shape == (2, d * g, 6, 8)
    diff = float((a - b).abs().max())
    assert diff < 1e-5, f"dense conv differs: maxdiff {diff:.3e}"
    print(f"  ok  dense conv == grouped conv: maxdiff {diff:.3e}")


def test_converted_model_matches_prered_and_original():
    """End-to-end disparity: dense == prered (tight) and dense == original."""
    orig = _random_both(seed=0)
    prered = prered_from_original(orig).eval()
    dense = convert_from_prered(prered).eval()
    dense2 = convert_from_original(orig).eval()
    torch.manual_seed(7)
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    with torch.no_grad():
        a = orig(left, right)
        p = prered(left, right)
        b = dense(left, right)
        c = dense2(left, right)
    assert b.shape == (1, 1, 64, 128), b.shape
    dp = float((p - b).abs().max())
    do = float((a - b).abs().max())
    dc = float((b - c).abs().max())
    assert dp < 1e-5, f"dense vs prered maxdiff {dp:.3e} px too large"
    assert do < 1e-4, f"dense vs original maxdiff {do:.3e} px too large"
    assert dc == 0.0, f"two dense paths differ: {dc:.3e}"
    print(f"  ok  dense==prered {dp:.3e} px, dense==orig {do:.3e} px")


def test_output_shape_and_channel_counts():
    """Shape contract: cost volume [B, 2*D*G, h, w], disparity [B, 1, H, W];
    dense fused_corr [D*G, D*C, 1, 1] groups=1; reduce unchanged."""
    torch.manual_seed(0)
    m = convert_from_original(_random_both()).eval()
    assert isinstance(m, HailoStereoDense)
    assert isinstance(m.cost_volume, DenseCorrCostVolume)
    assert m.cost_volume.out_channels == 2 * NUM_DISP * 8 == 384
    assert m.cost_volume.reduce.weight.shape == (8, 4, 1, 1)
    assert m.cost_volume.fused_corr.weight.shape == (192, 768, 1, 1)
    assert m.cost_volume.fused_corr.groups == 1
    assert m.aggregation.stem[0].weight.shape[1] == 384
    # Prered stem must be reused unchanged (same layout).
    prered = prered_from_original(_random_both(seed=0)).eval()
    torch.manual_seed(0)
    m2 = convert_from_prered(prered).eval()
    assert torch.equal(m2.aggregation.stem[0].weight,
                       prered.aggregation.stem[0].weight)
    torch.manual_seed(2)
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128))
    assert out.shape == (1, 1, 64, 128), out.shape
    assert torch.isfinite(out).all()
    print("  ok  shapes: fused_corr [192,768,1,1] groups=1, disparity [1,1,64,128]")


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
    """Only cost='both' (the EdgeDepth-8 arch) can be densified."""
    torch.manual_seed(0)
    try:
        HailoStereoDense(cost="sub")
    except AssertionError:
        print("  ok  HailoStereoDense(cost='sub') refused")
    else:
        raise AssertionError("HailoStereoDense(cost='sub') should assert")
    try:
        convert_from_original(_random_both(cost="sub"))
    except AssertionError:
        print("  ok  convert_from_original(cost='sub') refused")
    else:
        raise AssertionError("convert_from_original(cost='sub') should assert")


if __name__ == "__main__":
    for fn in (test_dense_conv_equals_grouped_conv,
               test_converted_model_matches_prered_and_original,
               test_output_shape_and_channel_counts,
               test_stem_perm_is_consistent,
               test_rejects_non_both_cost):
        fn()
    print("dense: all tests passed")
