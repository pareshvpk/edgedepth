#!/bin/bash
# Stage 2 after screen_queue.sh: two independent finetunes of runs/kft_both/best.pt.
#  kft_both_n30: kitti_ft_near30 recipe -- old arch 0-5 m EPE 6.77 -> 5.39, masked ~flat
#  kft_both_s60: kitti_ft_scale60 recipe -- old arch masked 1.156 -> 1.122, 0-5 m 7.10 -> 6.74
# Pick the final model by masked EPE under target AND best 0-5 m (eval_depth).
# Thermal: --temp-target 68 governor + 75 C watchdog (15 min SIGSTOP), as screen_queue.sh.
cd "$(dirname "$0")/.."
PY=~/.venvs/stereo/bin/python
COMMON="--dataset kitti_mixed --root data/kitti2015/training --kitti2012-root data/kitti2012/training --init runs/kft_both/best.pt --batch 4 --ema 0.999 --aug strong --crop-h 320 --crop-w 768 --workers 8 --epoch-pause 10 --step-pause 0.1 --temp-target 68"
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
run() {  # name, extra flags...
  name=$1; shift
  while [ "$(temp)" -gt 75 ]; do echo "[stage2] GPU $(temp)C, cooling 15 min before $name"; sleep 900; done
  echo "[queue] start $name $(date)"
  $PY src/train.py $COMMON --out runs/$name "$@" >> runs/${name}_train.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    t=$(temp)
    if [ "$t" -gt 75 ]; then
      echo "[queue] GPU ${t}C: pausing $name for 15 min $(date +%T)"
      pkill -STOP -P $pid; kill -STOP $pid; sleep 900; kill -CONT $pid; pkill -CONT -P $pid
      echo "[queue] resumed $name at GPU $(temp)C $(date +%T)"
    fi
    sleep 5
  done
  wait $pid; echo "[queue] end $name rc=$? $(date)"
}
while pgrep -f '^/bin/bash runs/(screen|resume)_queue.sh' >/dev/null; do sleep 60; done
[ -f runs/kft_both/best.pt ] || { echo "[stage2] no runs/kft_both/best.pt, abort"; exit 1; }
run kft_both_n30 --epochs 30 --lr 3e-5 --near-weight 3 --scale-min 0.86 --scale-max 1.8 --scale-prob 0.6
run kft_both_s60 --epochs 60 --lr 5e-5 --scale-max 1.5 --scale-prob 0.5
