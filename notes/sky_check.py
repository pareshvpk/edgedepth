"""Sky / far-field behaviour that the KITTI val metric cannot see.
1) VKITTI2: EPE and mean prediction by GT band (sky, far, mid, near), 60 frames
   strided across all scenes/variants (NB: training draws from the same pool).
2) Real KITTI 2015 val: rows ABOVE the highest LiDAR return in each column --
   no GT there, so report the predicted depth distribution: share of those
   pixels the model places nearer than 20 m (sky/buildings should be far)."""
import sys, torch, numpy as np
sys.path.insert(0, 'src')
from model import HailoStereo, build_model
from data import VKitti2, Kitti2015

def load(p):
    b = torch.load(p, map_location='cpu', weights_only=False)
    m = build_model(b).cuda().eval()
    m.load_state_dict(b['model'])
    return m

vk = VKitti2('data/vkitti2', training=False); vk.samples = vk.samples[::len(vk.samples)//60][:60]
kv = Kitti2015('data/kitti2015/training', split='val')
FB = 386.48
for ck in sys.argv[1:]:
    m = load(ck); agg = {}
    with torch.no_grad():
        for s in vk:
            p = m(s['left'][None].cuda(), s['right'][None].cuda())[0,0].cpu().numpy()
            d = s['disp'][0].numpy(); e = np.abs(p-d); inr = (d>0)&(d<192)
            for name, sel in [('sky', inr&(d<0.6)), ('far', inr&(d>=0.6)&(d<10)),
                              ('mid', inr&(d>=10)&(d<75)), ('near', inr&(d>=75))]:
                a = agg.setdefault(name, [0,0,0]); a[0]+=e[sel].sum(); a[1]+=sel.sum(); a[2]+=p[sel].sum()
        near20 = tot = 0; preds = []
        for s in kv:
            p = m(s['left'][None].cuda(), s['right'][None].cuda())[0,0].cpu().numpy()
            d = s['disp'][0].numpy(); has = d > 0
            top = np.where(has.any(0), has.argmax(0), d.shape[0])     # first GT row per column
            rows = np.arange(d.shape[0])[:, None]
            above = rows < (top[None, :] - 10)                        # 10-row margin
            pa = p[above]; preds.append(pa)
            near20 += (pa > FB/20).sum(); tot += pa.size
    pa = np.concatenate(preds)
    print(f"\n== {ck}")
    print("  VKITTI2  " + "  ".join(f"{k}: EPE {v[0]/max(v[1],1):6.2f} pred {v[2]/max(v[1],1):5.1f}" for k,v in agg.items()))
    print(f"  KITTI above-LiDAR: median pred {np.median(pa):5.1f} px ({FB/max(np.median(pa),1e-3):5.1f} m), "
          f"{100*near20/tot:5.1f}% placed nearer than 20 m")
