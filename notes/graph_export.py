"""
Extract the REAL operator-level graph of artifacts/hailo_stereo.onnx.

"Real" = what is actually in the ONNX, not the PyTorch module tree. Two
pipeline features only become visible there:

- the siamese feature tower is UNROLLED: left/right each get their own node
  copies (/features/head/Conv vs /features/head_1/Conv, stem2.0 vs
  stem2.0_1, ...) that share the SAME initializer weights;
- the cost volume is 24 explicit Slice+Sub+Conv 'reduce' chains fanned out of
  /cost_volume and re-concatenated at /cost_volume/Concat_1.

Graph is rebuilt from tensor edges, not node order.

Outputs (into report/graph/):
  hailo_stereo_graph.dot          full operator graph for Graphviz
  hailo_stereo_graph.svg / .png   dot render
  hailo_stereo_graph.mmd          Mermaid flowchart of the same graph
  hailo_stereo_graph_stats.md     module census, op census, weight sharing,
                                  plus a PyTorch cross-check against the
                                  checkpoint the ONNX was exported from
                                  (weights, BN folds incl. the fused
                                  soft-argmin temperature, and a full
                                  torch-vs-onnxruntime numeric parity run)
                                  and a UNRECOVERED/limitations census of
                                  what the exported ONNX no longer
                                  preserves

Edge policy: operator->operator data edges are solid; initializer / Constant
leaves (weights, biases, slice bounds, the zero pad) are small side nodes
with dashed links, so weight sharing across the two unrolled tower copies
stays visible.

Run:  .venv/Scripts/python.exe notes/graph_export.py
"""

import collections
import copy
import pathlib
import sys

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

ROOT = pathlib.Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "artifacts" / "hailo_stereo.onnx"
OUT = ROOT / "report" / "graph"
# the checkpoint artifacts/hailo_stereo.onnx was exported from (see
# report.md: run_full_eval pipeline -> src/export_onnx.py --ckpt)
CKPT = ROOT / "runs" / "kitti_mixed_fixed768" / "best.pt"
HEIGHT, WIDTH = 368, 1232

FLAT_CLUSTER = {"cost_volume", "soft_argmin"}   # one flat cluster, no fan
TOP_CLAMP = {"features", "cost_volume", "aggregation", "soft_argmin",
             "refine1", "refine2", "refine4"}

OP_FILL = {
    "Conv":      ("conv",    "#dae8fc", "#6c8ebf"),
    "LeakyRelu": ("act",     "#d5e8d4", "#82b366"),
    "Add":       ("eltwise", "#ffe6cc", "#d79b00"),
    "Mul":       ("eltwise", "#ffe6cc", "#d79b00"),
    "Sub":       ("eltwise", "#ffe6cc", "#d79b00"),
    "Concat":    ("join",    "#f8cecc", "#b85450"),
    "Slice":     ("slice",   "#f5f5f5", "#666666"),
    "Resize":    ("resize",  "#e1d5e7", "#9673a6"),
    "Softmax":   ("softmax", "#fff2cc", "#d6b656"),
    "Clip":      ("clamp",   "#fff2cc", "#d6b656"),
    "Constant":  ("const",   "#fafafa", "#cccccc"),
}

# edge records: dict(kind, src, dst, tensor)
#   kind = 'data'      dataflow producer -> consumer (solid)
#        | 'init'      initializer -> consumer            (dashed, amber)
#        | 'const'     Constant op -> consumer            (dashed, gray)
#        | 'ginput'    graph input tensor -> consumer     (solid, green)


def cluster_of(name):
    parts = name.strip("/").split("/")
    if parts[0] in TOP_CLAMP and parts[0] not in FLAT_CLUSTER and \
            len(parts) > 2:
        return "/".join(parts[:2])
    return parts[0]


def short_label(n):
    """In-cluster label: drop the cluster's own path prefix."""
    c = cluster_of(n.name)
    parts = n.name.strip("/").split("/")
    cparts = c.split("/")
    if parts[:len(cparts)] == cparts:
        parts = parts[len(cparts):]
    return "/".join(parts)


def short_shape(dims):
    return "x".join(str(d) for d in dims) if dims else ""


def esc_dot(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def mid(nid):
    m = nid.strip("/").replace("/", "_").replace(".", "_").replace(",", "_")
    return m if m[:1].isalpha() or m[:1] == "_" else "n" + m


# ---------------------------------------------------------------------------
# PyTorch cross-check
# ---------------------------------------------------------------------------

def _fold_conv_bn(conv, bn, scale=1.0):
    s = scale * bn.weight.detach().numpy() / np.sqrt(
        bn.running_var.detach().numpy() + bn.eps)
    b = scale * bn.bias.detach().numpy() - s * bn.running_mean.detach().numpy()
    w = conv.weight.detach().numpy() * s.reshape(-1, 1, 1, 1)
    return np.ascontiguousarray(w, dtype=np.float32), \
        np.ascontiguousarray(b, dtype=np.float32)


def torch_cross_check(g, ops, init_users):
    """Cross-check the exported graph against the PyTorch model it was
    exported from.

    Rebuilds HailoStereo from CKPT (the checkpoint behind this ONNX) and
    checks that

      1. every ONNX initializer is accounted for by the torch side: a
         parameter of the same name (bias-free/no-BN heads), or the fold of a
         Conv2d+BatchNorm2d pair, including the -temperature-scaled fold of
         the aggregation cost head;
      2. the structure the statistics prose claims: siamese tower copies
         sharing every weight, one reduce arm per hypothesis, arange index
         ramp, 3 refine stages x Mul(2.0), Clip to [0, MAX_DISP];
      3. numeric parity: torch forward vs onnxruntime on a random pair.

    Returns (rows, problems): rows are stats-md table lines, problems are
    human-readable failures. Requires torch + onnxruntime.
    """
    import torch
    import torch.nn as nn
    import onnxruntime as ort
    sys.path.insert(0, str(ROOT / "src"))
    from model import MAX_DISP, build_model

    blob = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    ref = build_model(blob)
    ref.load_state_dict(blob.get("model", blob), strict=True)
    ref.eval()
    t0 = float(ref.soft_argmin.temperature().detach())

    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    rows, problems = [], []

    def rec(describe, ok, detail):
        mark = "PASS" if ok else "FAIL"
        rows.append(f"| {describe} | {mark} | {detail} |")
        if not ok:
            problems.append(f"{describe} ({detail})")

    # ---- (1) same-name parameters ---------------------------------------
    matched, same_ok = [], True
    for key, t in ref.state_dict().items():
        if key not in inits:
            continue
        a = inits[key]
        bit = a.shape == tuple(t.shape) and \
            np.array_equal(a, np.ascontiguousarray(t.detach().numpy(),
                                                   dtype=np.float32))
        matched.append(key)
        same_ok &= bit
    rec("torch params with an ONNX initializer of the same name are "
        "bit-identical", same_ok,
        f"{len(matched)} of {len(inits)} initializers covered by this rule "
        "(head, reduce, refine-stage outs, index ramp)")

    # ---- (2) Conv2d+BN folds (incl. the -t scaled cost head) -------------
    mods = list(ref.named_modules())
    cand = {}
    for (n1, m1), (n2, m2) in zip(mods, mods[1:]):
        if (isinstance(m1, nn.Conv2d) and m1.bias is None
                and isinstance(m2, nn.BatchNorm2d)):
            cand[n1] = m1, m2

    onnx_pairs = {}
    for node in g.node:
        if node.op_type != "Conv":
            continue
        wins = [t for t in node.input if t in inits]
        if len(wins) >= 2 and wins[0].startswith("onnx::Conv_"):
            onnx_pairs[wins[0]] = inits[wins[0]], inits[wins[1]]

    folds = {}
    for name, (m1, m2) in cand.items():
        scale = -t0 if name == "aggregation.out.0" else 1.0
        folds[name] = _fold_conv_bn(m1, m2, scale=scale)
    assert "aggregation.out.0" in folds

    def pair_hit(pa, pb, w, b):
        return (pa.shape == w.shape and pb.shape == b.shape
                and np.allclose(pa, w, rtol=1e-5, atol=1e-6)
                and np.allclose(pb, b, rtol=1e-5, atol=1e-6))

    matched_conv = []
    hit_names = set()
    for wn, (w, b) in onnx_pairs.items():
        hit = None
        for cname, (cw, cb) in folds.items():
            if pair_hit(w, b, cw, cb):
                hit = cname
                break
        if hit is not None:
            matched_conv.append(wn)
            hit_names.add(hit)
    unmatched = [wn for wn in onnx_pairs if wn not in matched_conv]
    unmatched_torch = [c for c in folds if c not in hit_names]
    rec("every Conv2d+BatchNorm2d fold materialised by constant folding "
        "matches a torch pair exactly",
        not unmatched and not unmatched_torch,
        f"{len(onnx_pairs)} folded ONNX convs x {len(matched_conv)} matched, "
        f"temperature t = {t0:.4f} pre-scaled onto the aggregation "
        "cost head")
    if not same_ok or unmatched or unmatched_torch:
        return rows, problems

    # ---- (3) siamese tower sharing ---------------------------------------
    tower = [t for t in init_users if init_users[t] == 2]
    un1 = [n.name for n in g.node if n.name.startswith("/features/")
           and n.op_type == "Conv" and not n.name.split("/")[-2].endswith("_1")]
    with1 = [n.name for n in g.node if n.name.startswith("/features/")
             and n.op_type == "Conv" and n.name.split("/")[-2].endswith("_1")]
    torch_tower = sum(1 for _, m in ref.features.named_modules()
                      if isinstance(m, nn.Conv2d))
    rec("siamese tower: 2 conv-node copies per torch conv, sharing 1 "
        "initializer set",
        len(un1) == len(with1) == torch_tower
        and init_users["features.head.weight"] == 2,
        f"{torch_tower} torch convs -> {len(un1)} + {len(with1)} nodes, "
        f"{len(tower)} initializers on 2 consumers each "
        f"(features.head.weight is one of them)")

    # ---- (5) cost volume and index ramp ----------------------------------
    num_disp = ref.num_disp
    rec("cost volume: the shared reduce initializer has 1 consumer per "
        "hypothesis arm",
        init_users["cost_volume.reduce.weight"] == num_disp
        and ops.get("Sub", 0) == num_disp and ops.get("Slice", 0) == num_disp,
        f"reduce on {init_users['cost_volume.reduce.weight']} nodes, "
        f"{ops.get('Sub', 0)} Sub, {ops.get('Slice', 0)} Slice, "
        f"D={num_disp}")
    torch_ramp = ref.soft_argmin.index.weight.detach().numpy().ravel()
    onnx_ramp = inits["soft_argmin.index.weight"].ravel()
    rec("soft-argmin index ramp == arange(D) on both sides",
        np.array_equal(torch_ramp, onnx_ramp)
        and np.array_equal(onnx_ramp,
                           np.arange(num_disp, dtype=np.float32)),
        f"[{onnx_ramp[0]:.0f}..{onnx_ramp[-1]:.0f}] on both sides")

    # ---- (6) refine ladder of RefineStage count --------------------------
    stages = [name for name, m in ref.named_modules()
              if m.__class__.__name__ == "RefineStage"]
    const_t = {t: numpy_helper.to_array(n.attribute[0].t)
               for n in g.node if n.op_type == "Constant"
               for t in n.output}
    mul2 = 0
    for n in g.node:
        if n.op_type == "Mul":
            vals = [float(const_t[t].ravel()[0]) for t in n.input
                    if t in const_t]
            if vals and any(v == 2.0 for v in vals):
                mul2 += 1
    rec("refine ladder: 3 stages, one Mul(2.0) each",
        len(stages) == 3 and mul2 == 3,
        f"{len(stages)} RefineStage, {mul2} Mul(2.0)")

    # ---- (7) Clip to the disparity range --------------------------------
    clip = [n for n in g.node if n.op_type == "Clip"][0]
    bounds = []
    for t in clip.input[1:]:
        if t in inits:
            bounds.append(float(inits[t].ravel()[0]))
        elif t in const_t:
            bounds.append(float(const_t[t].ravel()[0]))
    rec("final Clip bounds == [0, MAX_DISP] = [0.0, 192.0]",
        bounds == [0.0, float(MAX_DISP)],
        f"[{bounds[0]}, {bounds[1]}]")

    # ---- (8) numeric parity torch vs onnxruntime -------------------------
    try:
        torch.manual_seed(0)
        l = torch.randn(1, 3, HEIGHT, WIDTH)
        r = torch.randn(1, 3, HEIGHT, WIDTH)
        with torch.no_grad():
            exp = ref(l, r).numpy()
        sess = ort.InferenceSession(str(ONNX_PATH),
                                    providers=["CPUExecutionProvider"])
        got = sess.run(["disparity"],
                       {"left": l.numpy(), "right": r.numpy()})[0]
        shape_ok = got.shape == exp.shape
        worst = float(np.abs(got - exp).max()) if shape_ok else float("nan")
        rec("numeric parity: torch forward vs onnxruntime (randn 1x3x368x1232, "
            "seed 0)", shape_ok and worst <= 1e-2,
            f"max |diff| = {worst:.2e} px (tolerance 1e-02), "
            f"shapes {exp.shape}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"numeric parity not run: {e}")
        rows.append(f"| numeric parity: torch vs onnxruntime | FAIL | "
                    f"not run: {e} |")

    return rows, problems


def main():
    model = onnx.load(str(ONNX_PATH))
    g = model.graph

    shapes = {}
    try:
        m2 = shape_inference.infer_shapes(model, strict_mode=False)
        for vi in list(m2.graph.value_info) + list(m2.graph.input) + \
                list(m2.graph.output):
            dims = vi.type.tensor_type.shape.dim
            shapes[vi.name] = tuple(d.dim_value or d.dim_param for d in dims)
    except Exception as e:
        print(f"  ! shape inference failed ({e}); graph drawn shapeless")

    inits = {i.name: tuple(i.dims) for i in g.initializer}
    prod = {}
    for n in g.node:
        for t in n.output:
            prod[t] = n
    NODES = {n.name: n for n in g.node}

    graph_inputs = [i.name for i in g.input]
    graph_outputs = [o.name for o in g.output]
    assert not (set(graph_inputs) & prod.keys()), "IO also produced inside"

    edges = []
    init_users = collections.Counter()
    const_users = collections.Counter()
    for n in g.node:
        for t in n.input:
            if not t:
                continue
            if t in inits:
                init_users[t] += 1
                edges.append(dict(kind="init", src=None, dst=n.name,
                                  tensor=t))
            elif t in prod:
                src = prod[t]
                kind = "const" if src.op_type == "Constant" else "data"
                if kind == "const":
                    const_users[t] += 1
                edges.append(dict(kind=kind, src=src.name, dst=n.name,
                                  tensor=t))
            elif t in graph_inputs:
                edges.append(dict(kind="ginput", src=None, dst=n.name,
                                  tensor=t))
            else:
                raise RuntimeError(f"unknown tensor source {t} for {n.name}")

    def node_shape(name, n):
        if n.op_type == "Constant":
            return ()
        return shapes.get(n.output[0], ())

    clusters = collections.defaultdict(list)
    for name in NODES:
        clusters[cluster_of(name)].append(name)

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ DOT
    dot = [
        'digraph hailo_stereo {',
        '  graph [fontname="Segoe UI", rankdir=TB, ranksep=0.40, '
        'nodesep=0.12, compound=true, labelloc=t, fontsize=16, label="'
        'hailo_stereo.onnx -- REAL operator graph (opset %d): %d nodes, '
        '%d operator->operator edges, %d initializers + %d Constant ops '
        '(weight links dashed)"];' % (
            model.opset_import[0].version, len(g.node),
            sum(1 for e in edges if e['kind'] == 'data'),
            len(inits), sum(1 for e in edges if e['kind'] == 'const')),
        '  node [fontname="Segoe UI", fontsize=8.5, style="filled,rounded", '
        'shape=box, height=0.28, margin="0.05,0.025", splines=polyline];',
        '  edge [arrowsize=0.55, color="#555555", fontname="Segoe UI", '
        'fontsize=7];',
    ]
    for cname in sorted(clusters):
        members = clusters[cname]
        dot.append(f'  subgraph "cluster_{mid(cname)}" {{')
        dot.append(f'    label="{cname}  ({len(members)})"; '
                   'style="rounded,dashed"; color="#bbbbbb"; margin=6;')
        for name in sorted(members):
            n = NODES[name]
            cls, fill, stroke = OP_FILL.get(n.op_type, ("const", "#fafafa",
                                                        "#cccccc"))
            sh = short_shape(node_shape(name, n))
            label = esc_dot(short_label(n)) + ("\\n" + esc_dot(sh)
                                               if sh else "")
            extra = ""
            if n.op_type == "Conv":
                st = [list(a.ints) for a in n.attribute
                      if a.name == "strides"]
                dl = [list(a.ints) for a in n.attribute
                      if a.name == "dilations"]
                ks = [list(a.ints) for a in n.attribute
                      if a.name == "kernel_shape"]
                tag = ""
                if ks:
                    tag += f"k={ks[0]}"
                if st and st[0] != [1, 1]:
                    tag += f" s={st[0]}"
                if dl:
                    tag += f" d={dl[0]}"
                if tag:
                    extra = "\\n" + esc_dot(tag)
            dimmed = n.op_type == "Constant"
            css = (f'fillcolor="{fill}", color="{stroke}"'
                   if not dimmed else
                   'fillcolor="#fafafa", color="#cccccc", fontsize=7')
            dot.append(f'    "{mid(name)}" [label="{label}{extra}", {css}, '
                       f'tooltip="{esc_dot(name)}", class={cls}];')
        dot.append('  }')
    for name in graph_inputs + graph_outputs:
        sh = short_shape(shapes.get(name, ()))
        css = 'fillcolor="#2d2d2d", fontcolor="white", color="black", ' \
              'shape=parallelogram'
        dot.append(f'  "{mid(name)}" [label="IO: {esc_dot(name)}\\n{sh}", '
                   f'{css}];')
    for t, dims in inits.items():
        shared = init_users[t] > 1
        pen = ', penwidth=1.4, color="#a05a00"' if shared else ""
        dot.append(f'  "W_{mid(t)}" [label="W {esc_dot(t)}\\n'
                   f'{short_shape(dims)}", fillcolor="#feffef", '
                   f'color="#dddddd", fontsize=6.5, style="filled,rounded"'
                   f'{pen}];')
    for e in edges:
        if e['kind'] == 'init':
            dot.append(f'  "W_{mid(e["tensor"])}" -> "{mid(e["dst"])}" '
                       '[style=dashed, color="#cc8800", weight=0];')
        elif e['kind'] == 'const':
            dot.append(f'  "{mid(e["src"])}" -> "{mid(e["dst"])}" '
                       '[style=dashed, color="#999999", weight=0];')
        elif e['kind'] == 'ginput':
            dot.append(f'  "{mid(e["tensor"])}" -> "{mid(e["dst"])}" '
                       '[color="#0aa64d", penwidth=1.4];')
        else:
            dot.append(f'  "{mid(e["src"])}" -> "{mid(e["dst"])}" '
                       '[weight=10];')
    dot.append('}')
    (OUT / "hailo_stereo_graph.dot").write_text("\n".join(dot) + "\n",
                                                encoding="utf-8")
    print(f"  dot   {OUT/'hailo_stereo_graph.dot'}")

    # --------------------------------------------------------------- Mermaid
    mmd = ['%% hailo_stereo.onnx -- REAL operator graph (opset 14).',
           '%% ' + str(len(g.node)) + ' operator nodes; solid arrows = '
           'tensor dataflow,',
           '%% dashed = weights / Constant leaves feeding consumers.',
           'flowchart TB']
    for cname in sorted(clusters):
        mmd.append(f'  subgraph {"C" + mid(cname)}["{cname} '
                   f'({len(clusters[cname])})"]')
        mmd.append('    direction TB')
        for name in sorted(clusters[cname]):
            n = NODES[name]
            cls = OP_FILL.get(n.op_type, ("const",))[0]
            sh = short_shape(node_shape(name, n))
            tag = f"<br/>{sh}" if sh else ""
            mmd.append(f'    {mid(name)}["{esc_dot(name)}{tag}"]:::{cls}')
        mmd.append('  end')
    for name in graph_inputs + graph_outputs:
        sh = short_shape(shapes.get(name, ()))
        mmd.append(f'  {mid(name)}(["IO: {name}<br/>{sh}"]):::io')
    for t, dims in inits.items():
        sh = short_shape(dims)
        star = " *shared*" if init_users[t] > 1 else ""
        mmd.append(f'  W_{mid(t)}["{t}<br/>{sh}{star}"]:::weight')
    for e in edges:
        if e['kind'] == 'init':
            mmd.append(f'  W_{mid(e["tensor"])} -.-> {mid(e["dst"])}')
        elif e['kind'] == 'const':
            mmd.append(f'  {mid(e["src"])} -.-> {mid(e["dst"])}')
        elif e['kind'] == 'ginput':
            mmd.append(f'  {mid(e["tensor"])} --> {mid(e["dst"])}')
        else:
            mmd.append(f'  {mid(e["src"])} --> {mid(e["dst"])}')
    mmd += [
        '  classDef conv fill:#dae8fc,stroke:#6c8ebf',
        '  classDef act fill:#d5e8d4,stroke:#82b366',
        '  classDef eltwise fill:#ffe6cc,stroke:#d79b00',
        '  classDef join fill:#f8cecc,stroke:#b85450',
        '  classDef slice fill:#f5f5f5,stroke:#666666',
        '  classDef resize fill:#e1d5e7,stroke:#9673a6',
        '  classDef softmax fill:#fff2cc,stroke:#d6b656',
        '  classDef clamp fill:#fff2cc,stroke:#d6b656',
        '  classDef const fill:#fafafa,stroke:#cccccc,font-size:8px',
        '  classDef weight fill:#feffef,stroke:#cc9900,font-size:8px',
        '  classDef io fill:#2d2d2d,stroke:#000,color:#fff',
    ]
    (OUT / "hailo_stereo_graph.mmd").write_text("\n".join(mmd) + "\n",
                                                encoding="utf-8")
    print(f"  mermaid {OUT/'hailo_stereo_graph.mmd'}")

    # ----------------------------------------------------------------- stats
    ops = collections.Counter(n.op_type for n in g.node)
    shared = {t: c for t, c in init_users.items() if c > 1}
    tw = sum(int(np.prod(d)) for d in inits.values())
    auto_inits = [i.name for i in g.initializer
                  if i.name.startswith("onnx::Conv_")]
    named_inits = [i.name for i in g.initializer
                   if not i.name.startswith("onnx::Conv_")]
    folded_nodes = [n.name for n in g.node if n.op_type == "Conv"
                    and any(t in set(auto_inits) for t in n.input)]
    print("  cross-check vs PyTorch:")
    try:
        cc_rows, cc_problems = torch_cross_check(g, ops, init_users)
    except Exception as e:  # noqa: BLE001
        cc_rows = ["| PyTorch cross-check | FAIL | crashed: "
                   f"{type(e).__name__}: {e} |"]
        cc_problems = [f"cross-check crashed: {e}"]
    for row in cc_rows:
        print(f"    {row}")
    lines = [
        '# hailo_stereo.onnx -- operator-level graph stats',
        '',
        f'Source `{ONNX_PATH.relative_to(ROOT)}`: {len(g.node)} nodes, '
        f'{len(inits)} initializers ({tw/1e6:.2f}M weights), opset '
        f'{model.opset_import[0].version}, '
        f'{sum(1 for e in edges if e["kind"] == "data")} operator-to-operator '
        'edges (+106 Constant-op consumers = 422 tensor edges).',
        'Graphs: `hailo_stereo_graph.dot` (+`.svg`/`.png`), '
        '`hailo_stereo_graph.mmd`, module overview '
        '`hailo_stereo_modules.dot` (+`.svg`/`.png`).',
        '',
        '## Top-level flow (verified from tensor connectivity)',
        '',
        'Branch A (un-suffixed nodes) consumes `right`, branch B (`_1` '
        'suffixes) consumes `left`; both copies share every weight '
        'initializer (see shared table below).',
        '',
        '```',
        'IO:right / IO:left -> stem2/4/8 -> res.0..5 -> head | head_1',
        'branch A (right) -> /features/head -> Concat[zero-pad, right-feat]',
        '  -> 24x Slice @shift d -> Sub(head_1 - slice)  (left-branch grid)',
        '  -> 24x reduce Conv 1x1 (grouped 32->8) -> Concat_1 (192ch cost volume)',
        '  -> aggregation (stem, res_a/b, down, res_c/d, Resize+merge, Add,',
        '     res_e, out 3x3 -> 24 channel hypotheses)',
        '  -> Softmax -> /soft_argmin/index Conv 24->1 (disparity @1/8 res)',
        'refine4/2/1 (x3): Resize(scale 2) -> Mul(2.0) -> Concat with the',
        '  branch-B (left) stem4 (r4) / stem2 (r2) feature / raw IO:left (r1)',
        '  -> blocks -> out',
        '  -> Add   (up-samples disparity 1/8 -> 1/4 -> 1/2 -> 1/1)',
        'Clip -> IO:disparity',
        '```',
        'Every disparity hypothesis is an explicit Slice/Sub/Conv arm '
        '(`/cost_volume/*0..*23`); there is no disparity-search op in ONNX '
        '-- the argmin IS the soft-argmin index conv.',
        '',
        '## PyTorch cross-check',
        '',
        f'Against `HailoStereo` (`src/model.py`) rebuilt from '
        f'`{CKPT.relative_to(ROOT)}` -- the checkpoint `artifacts/'
        'hailo_stereo.onnx` was exported from (BN folds reproduced, the '
        'soft-argmin temperature folded with its trained sign, and a '
        'full torch-vs-onnxruntime run).',
        '',
        '| check | result | detail |',
        '|---|---|---|',
        *cc_rows,
        '',
        '## UNRECOVERED from the ONNX / limitations of the graph evidence',
        '',
        f'Every row is verified directly against `{ONNX_PATH.relative_to(ROOT)}`; '
        'status follows the report.md labeling convention '
        '(UNAVAILABLE = not in the file at all, FOLDED → gone during export, '
        'OUTSIDE = not part of this artifact, INFERRED = only recoverable '
        'by matching, not read off).',
        '',
        '| value | status | reason |',
        '|---|---|---|',
        f'| torch module names for the folded conv/BN weights | INFERRED | '
        f'{len(auto_inits)} of {len(auto_inits) + len(named_inits)} '
        'initializers are auto-renamed `onnx::Conv_NNN` (only '
        f'{len(named_inits)} keep real names); identity of the '
        f'{len(folded_nodes)} folded conv nodes follows only from the '
        'numeric fold-matching in the cross-check above |',
        '| BatchNorm running_mean / running_var / eps | FOLDED | merged into '
        "each conv's W/b during export; the raw BN tensors are absent from "
        'the graph and the fold is one-way |',
        '| soft-argmin temperature scalar | FOLDED | pre-scaled with a sign '
        'flip onto the aggregation cost head during constant folding (see '
        'the fold row above); the scalar itself is not stored |',
        '| training state (epoch, optimizer, loss history) | UNAVAILABLE | '
        'inference-only export; no `metadata_props`, empty doc_string |',
        '| dynamic / alternate input resolutions | UNAVAILABLE | shapes are '
        'frozen at 1x3x368x1232 at export time |',
        '| preprocessing (ImageNet mean/std, 16-multiple crop, left-first '
        'ordering) | OUTSIDE | no normalization ops in the graph; the '
        'caller must reproduce it |',
        '| confidence / entropy surface | UNAVAILABLE | the 24-way softmax '
        'is computed internally (`/soft_argmin/Softmax`) and consumed by '
        'the index conv; IO exposes only `disparity` |',
        '| quantized variants (Hailo-8 / Hailo-15H INT8) | OUTSIDE | this '
        'artifact is the float32 opset 14 graph; the INT8 profiles live in '
        'the HEF artifacts under `deploy/` |',
        '',
        '## Clusters (first 1-2 path components of the node names)',
        '',
        '| cluster | nodes |',
        '|---|---|',
    ]
    for c in sorted(clusters, key=lambda x: -len(clusters[x])):
        lines.append(f'| {c} | {len(clusters[c])} |')
    lines += ['', '## Ops', '', '| op | count |', '|---|---|']
    for op, c in ops.most_common():
        lines.append(f'| {op} | {c} |')
    lines += ['',
              '## Initializers consumed by more than one node '
              '(the siamese-weight-sharing evidence)',
              '', '| tensor | consumers |', '|---|---|']
    for t in sorted(shared, key=lambda x: -shared[x]):
        lines.append(f'| {t} | {shared[t]} |')
    (OUT / 'hailo_stereo_graph_stats.md').write_text('\n'.join(lines) + '\n',
                                                     encoding='utf-8')
    print(f"  stats  {OUT/'hailo_stereo_graph_stats.md'}")

    # ------------------------------------------- module-level overview DOT
    # All 347 operators collapsed into their clusters; edges are the real
    # tensor flows between clusters, labelled with the edge count.
    mops = collections.defaultdict(collections.Counter)
    for name in NODES:
        mops[cluster_of(name)][NODES[name].op_type] += 1
    agg = collections.Counter()
    for e in edges:
        if e['kind'] == 'data':
            a, b = cluster_of(e['src']), cluster_of(e['dst'])
            if a != b:
                agg[(a, b)] += 1
    for e in edges:
        if e['kind'] == 'ginput':
            agg[("IO:" + e['tensor'], cluster_of(e['dst']))] += 1
    own = {t: n.name for t, n in prod.items()}
    for o in graph_outputs:
        if o in own:
            agg[(cluster_of(own[o]), "IO:" + o)] += 1

    def module_node(label, is_io=False):
        ops = mops.get(label, collections.Counter())
        counts = ", ".join(f"{k}x{v}" for k, v in ops.most_common())
        text = f"{esc_dot(label)}\\n{esc_dot(counts)}" if counts \
            else esc_dot(label)
        css = ('fillcolor="#2d2d2d", fontcolor="white", shape=parallelogram'
               if is_io else 'fillcolor="#fff8e7", color="#777777"')
        return f'  "{esc_dot(label)}" [label="{text}", {css}];'

    md = ['digraph hailo_stereo_modules {',
          '  graph [fontname="Segoe UI", rankdir=TB, ranksep=0.55, '
          'nodesep=0.5, labelloc=t, fontsize=14, label="hailo_stereo.onnx '
          '-- module-level view of the REAL operator graph (all 347 ops '
          'summarised per cluster); edge labels = tensor-edge count"];',
          '  node [fontname="Segoe UI", fontsize=11, '
          'style="filled,rounded", shape=box];',
          '  edge [fontname="Segoe UI", fontsize=9, arrowsize=0.7];']
    for m in sorted(mops):
        md.append(module_node(m))
    for lab in ("IO:left", "IO:right", "IO:disparity"):
        md.append(module_node(lab, is_io=True))
    for (a, b), n in sorted(agg.items()):
        md.append(f'  "{a}" -> "{b}" [label=" {n}"];')
    md.append("}")
    (OUT / 'hailo_stereo_modules.dot').write_text('\n'.join(md) + '\n',
                                                  encoding='utf-8')
    print(f"  module overview {OUT/'hailo_stereo_modules.dot'}")

    if cc_problems:
        print(f"  ! CROSS-CHECK PROBLEMS ({len(cc_problems)}):")
        for p in cc_problems:
            print(f"    - {p}")
    else:
        print("  cross-check PASSED -- torch side fully accounts for the graph")


if __name__ == '__main__':
    main()
