"""
Regression tests for the stereo matching path.

The Hailo Model Zoo `stereonet` shipped for years with a cost volume whose 12
disparity hypotheses were bit-identical -- it performed no disparity search at
all. A shape-only smoke test passes happily on that model. These tests do not.

Run directly:   python src/test_disparity.py
Or via pytest:  pytest src/test_disparity.py -v
"""

from __future__ import annotations

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import (  # noqa: E402
    HailoStereo,
    CostVolume,
    shift_right_features,
    _shift_all,
    MATCH_SCALE,
    NUM_DISP,
)


def test_export_shift_matches_reference():
    """The export-friendly pad-once/slice-many form must be numerically
    identical to the reference shift. These are two implementations of the same
    equation and they are allowed to disagree by nothing at all."""
    x = torch.randn(1, 8, 5, 40)
    fast = _shift_all(x, NUM_DISP)
    assert len(fast) == NUM_DISP
    for d in range(NUM_DISP):
        ref = shift_right_features(x, d)
        assert torch.equal(fast[d], ref), \
            f"d={d}: static-graph shift disagrees with the reference shift"
    print(f"  ok  static shift matches reference for all {NUM_DISP} hypotheses")


# ---------------------------------------------------------------------------
# 1. the shift itself
# ---------------------------------------------------------------------------

def test_shift_actually_shifts():
    """out[..., x] must equal in[..., x-d], with the first d columns zeroed.

    This is the direct assertion the original model fails: its shift returned
    the input unchanged for every d.
    """
    w = NUM_DISP * 2 + 16          # wider than the largest hypothesis
    # each column holds its own index, so a shift is trivially readable
    x = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).repeat(1, 4, 3, 1)

    for d in range(NUM_DISP):
        out = shift_right_features(x, d)
        assert out.shape == x.shape, f"d={d}: shift changed the shape"
        if d > 0:
            assert torch.all(out[..., :d] == 0), \
                f"d={d}: columns before the shift boundary must be zero"
        moved = out[..., d:]
        expect = x[..., : w - d]
        assert torch.equal(moved, expect), \
            f"d={d}: shifted content is wrong -- this is the CV-1 defect"
    print(f"  ok  shift is a real shift for all d in 0..{NUM_DISP - 1}")


def test_shift_is_not_identity():
    """Explicitly reject the original's failure mode."""
    x = torch.randn(1, 8, 5, 32)
    identical = [d for d in range(1, NUM_DISP)
                 if torch.equal(shift_right_features(x, d), x)]
    assert not identical, \
        f"shift is a no-op at d={identical} -- the CV-1 defect has returned"
    print("  ok  no disparity level is a no-op")


# ---------------------------------------------------------------------------
# 2. the cost volume built from it
# ---------------------------------------------------------------------------

def test_cost_volume_slices_are_distinct():
    """The exact check that fails on the shipped stereonet, where every
    hypothesis reduced to `left - right` at zero disparity."""
    cv = CostVolume(ch=32, num_disp=NUM_DISP, groups=8).eval()
    left, right = torch.randn(1, 32, 24, 64), torch.randn(1, 32, 24, 64)

    with torch.no_grad():
        vol = cv(left, right)

    g = cv.groups
    assert vol.shape[1] == NUM_DISP * g
    base = vol[:, :g]
    collisions = []
    for d in range(1, NUM_DISP):
        sl = vol[:, d * g:(d + 1) * g]
        if torch.equal(sl, base):
            collisions.append(d)
    assert not collisions, \
        f"cost-volume levels {collisions} are identical to level 0 (CV-1)"
    print(f"  ok  all {NUM_DISP} cost-volume levels are distinct")


# ---------------------------------------------------------------------------
# 3. does matching actually find the right disparity?
# ---------------------------------------------------------------------------

def test_matching_finds_known_disparity():
    """End-to-end geometric check that needs no trained weights.

    Build a right image that is the left image shifted by exactly `k*8` pixels.
    Because the tower is stride-8, its 1/8 features are then shifted by exactly
    `k` cells, so the feature difference at hypothesis d=k is exactly zero --
    whatever the (random) weights happen to be. The argmin of the raw matching
    energy must therefore land on k.

    A model with a broken shift produces a flat energy profile and fails here.
    """
    torch.manual_seed(0)
    model = HailoStereo().eval()

    h, w, pad = 96, 512, MATCH_SCALE * NUM_DISP
    scene = torch.rand(1, 3, h, w + 2 * pad)

    failures = []
    for k in (0, 1, 3, 7, 12, 20):
        # A point at left column x appears at right column x - k*8, so the right
        # image is the scene sampled k*8 pixels further along: right[x] = left[x + k*8].
        shift_px = k * MATCH_SCALE
        left = scene[..., pad:pad + w]
        right = scene[..., pad + shift_px:pad + shift_px + w]

        with torch.no_grad():
            _, _, fl = model.features(left)
            _, _, fr = model.features(right)
            # crop off the border cells, where convolution padding breaks the
            # exact-shift relationship
            margin = NUM_DISP + 4
            energy = [
                (fl[..., margin:] - shift_right_features(fr, d)[..., margin:])
                .pow(2).mean().item()
                for d in range(NUM_DISP)
            ]

        best = min(range(NUM_DISP), key=energy.__getitem__)
        if best != k:
            failures.append((k, best, energy[k], energy[best]))
        else:
            runner_up = sorted(energy)[1]
            print(f"  ok  true disparity {shift_px:3d}px -> argmin bin {best:2d}"
                  f"   energy {energy[k]:.2e} vs next {runner_up:.2e}")

    assert not failures, (
        "matching did not recover the known disparity: "
        + "; ".join(f"true bin {k} -> got {g}" for k, g, _, _ in failures)
    )


def test_disparity_output_is_sane():
    """Shape, sign and finiteness of the deployed (eval-mode) output."""
    model = HailoStereo().eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 96, 320), torch.rand(1, 3, 96, 320))
    assert out.shape == (1, 1, 96, 320), f"unexpected output shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "output contains NaN or Inf"
    assert (out >= 0).all(), "disparity must be non-negative after the final ReLU"
    print(f"  ok  output {tuple(out.shape)}, range "
          f"[{out.min():.2f}, {out.max():.2f}] px")


# ---------------------------------------------------------------------------
# 4. refinement v2 (Hailo-safe, 1/8 matching unchanged)
# ---------------------------------------------------------------------------

def test_base_arch_keys_unchanged():
    """refine="base" must build EXACTLY today's graph: same state_dict keys
    and shapes as a HailoStereo() built with no args."""
    a = HailoStereo()
    b = HailoStereo(refine="base")
    ka, kb = list(a.state_dict().keys()), list(b.state_dict().keys())
    assert ka == kb, \
        f"base arch keys changed: only-in-base {[k for k in ka if k not in kb][:5]}, " \
        f"only-in-explicit {[k for k in kb if k not in ka][:5]}"
    for k in ka:
        assert tuple(a.state_dict()[k].shape) == tuple(b.state_dict()[k].shape), \
            f"base arch shape changed at {k}"
    print(f"  ok  refine=\"base\" keeps all {len(ka)} state_dict keys")


def test_v2_forward_shapes():
    """v2 in eval returns a single [1,1,H,W] tensor; in train a 4-item list."""
    model = HailoStereo(refine="v2").eval()
    left, right = torch.rand(1, 3, 256, 512), torch.rand(1, 3, 256, 512)
    with torch.no_grad():
        out = model(left, right)
    assert isinstance(out, torch.Tensor), f"eval must return a tensor, got {type(out)}"
    assert out.shape == (1, 1, 256, 512), f"unexpected eval shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "eval output contains NaN or Inf"
    model.train()
    with torch.no_grad():
        outs = model(left, right)
    assert isinstance(outs, list) and len(outs) == 4, \
        f"train must return a 4-item list, got {type(outs)}"
    print(f"  ok  v2 eval {tuple(out.shape)}, train {[tuple(o.shape) for o in outs]}")


def test_v2_fuse_parity():
    """fuse_temperature_() must not change v2 eval output (max abs diff < 1e-4)."""
    torch.manual_seed(0)
    model = HailoStereo(refine="v2").eval()
    left, right = torch.rand(1, 3, 96, 320), torch.rand(1, 3, 96, 320)
    with torch.no_grad():
        before = model(left, right)
    model.fuse_temperature_()
    with torch.no_grad():
        after = model(left, right)
    diff = (before - after).abs().max().item()
    assert diff < 1e-4, f"fused v2 output drifted by {diff:.2e} px"
    print(f"  ok  v2 fuse parity, max abs diff {diff:.2e} px")


# ---------------------------------------------------------------------------
# 5. refinement v3 (Hailo-safe 1/4 difference-only cost volume, no ew_mult)
# ---------------------------------------------------------------------------

# state_dict key counts recorded BEFORE the v3 edit (default constructor args)
BASE_KEY_COUNT = 286
V2_KEY_COUNT = 322


def test_base_v2_keys_unchanged():
    """refine="base"/"v2" graphs must be untouched by the v3 edit: hard-coded
    key counts plus key-set equality against a freshly built model each."""
    base = HailoStereo(refine="base")
    v2 = HailoStereo(refine="v2")
    kb, kv = list(base.state_dict().keys()), list(v2.state_dict().keys())
    assert len(kb) == BASE_KEY_COUNT, \
        f"base keys {len(kb)} != pre-v3 count {BASE_KEY_COUNT}"
    assert len(kv) == V2_KEY_COUNT, \
        f"v2 keys {len(kv)} != pre-v3 count {V2_KEY_COUNT}"
    assert kb == list(HailoStereo().state_dict().keys()), \
        "base key order changed"
    assert kv == list(HailoStereo(refine="v2").state_dict().keys()), \
        "v2 key order changed"
    assert not [k for k in kb + kv
                if k.split(".")[0] in ("match4", "cost_volume4", "cost4_proj")], \
        "v3-only modules leaked into base/v2"
    print(f"  ok  base {len(kb)} keys, v2 {len(kv)} keys, both unchanged")


def test_v3_forward_shapes():
    """v3 in eval returns a single [1,1,H,W] tensor; in train a 4-item list."""
    model = HailoStereo(refine="v3").eval()
    left, right = torch.rand(1, 3, 256, 512), torch.rand(1, 3, 256, 512)
    with torch.no_grad():
        out = model(left, right)
    assert isinstance(out, torch.Tensor), f"eval must return a tensor, got {type(out)}"
    assert out.shape == (1, 1, 256, 512), f"unexpected eval shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "eval output contains NaN or Inf"
    model.train()
    with torch.no_grad():
        outs = model(left, right)
    assert isinstance(outs, list) and len(outs) == 4, \
        f"train must return a 4-item list, got {type(outs)}"
    print(f"  ok  v3 eval {tuple(out.shape)}, train {[tuple(o.shape) for o in outs]}")


def test_v3_fuse_parity():
    """fuse_temperature_() must not change v3 eval output (max abs diff < 1e-4)."""
    torch.manual_seed(0)
    model = HailoStereo(refine="v3").eval()
    left, right = torch.rand(1, 3, 96, 320), torch.rand(1, 3, 96, 320)
    with torch.no_grad():
        before = model(left, right)
    model.fuse_temperature_()
    with torch.no_grad():
        after = model(left, right)
    diff = (before - after).abs().max().item()
    assert diff < 1e-4, f"fused v3 output drifted by {diff:.2e} px"
    print(f"  ok  v3 fuse parity, max abs diff {diff:.2e} px")


# ---------------------------------------------------------------------------

def main():
    tests = [
        ("shift is a real shift", test_shift_actually_shifts),
        ("shift is never identity", test_shift_is_not_identity),
        ("static shift matches reference", test_export_shift_matches_reference),
        ("cost volume levels distinct", test_cost_volume_slices_are_distinct),
        ("matching recovers known disparity", test_matching_finds_known_disparity),
        ("output is well formed", test_disparity_output_is_sane),
        ("base arch keys unchanged", test_base_arch_keys_unchanged),
        ("v2 forward shapes", test_v2_forward_shapes),
        ("v2 fuse parity", test_v2_fuse_parity),
        ("base/v2 keys unchanged", test_base_v2_keys_unchanged),
        ("v3 forward shapes", test_v3_forward_shapes),
        ("v3 fuse parity", test_v3_fuse_parity),
    ]
    failed = 0
    for title, fn in tests:
        print(f"\n[{title}]")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {exc}")
    print("\n" + "-" * 60)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
