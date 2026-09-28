# Scratch analysis: linear weight blend of two checkpoints that share an init.
# usage: blend.py A.pt B.pt w out.pt   -> out = (1-w)*A + w*B on the 'model' state dict
import sys, torch
a_p, b_p, w, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
A = torch.load(a_p, map_location="cpu", weights_only=False)
B = torch.load(b_p, map_location="cpu", weights_only=False)
assert A.get("arch") == B.get("arch"), "arch mismatch"
sa, sb = A["model"], B["model"]
assert sa.keys() == sb.keys()
m = {k: ((1 - w) * sa[k].float() + w * sb[k].float()).to(sa[k].dtype) if sa[k].is_floating_point() else sa[k]
     for k in sa}
torch.save({"model": m, "arch": A.get("arch"), "max_disp": A.get("max_disp"), "epoch": -1, "epe": float("nan"),
            "blend": {"a": a_p, "b": b_p, "w": w}}, out)
print(out)
