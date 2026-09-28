#!/bin/bash
# Architecture screen: 30-epoch SceneFlow pretrain per variant, same recipe as
# runs/sceneflow_fixed (hold-out EPE 3.604). Thermal rule: above 75 C the
# trainer is frozen (SIGSTOP) for 15 minutes, then continued (SIGCONT).
cd "$(dirname "$0")/.."
PY=~/.venvs/stereo/bin/python
COMMON="--dataset sceneflow --root data/driving --epochs 30 --batch 8 --lr 5e-4 --crop-h 256 --crop-w 512 --workers 8 --aug strong --ema 0.999 --epoch-pause 30 --step-pause 0.1 --temp-target 68"
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
run() {  # name, extra flags...
  name=$1; shift
  while [ "$(temp)" -gt 75 ]; do echo "[queue] GPU $(temp)C, cooling 15 min before $name"; sleep 900; done
  echo "[queue] start $name $(date)"
  $PY src/train.py $COMMON --out runs/$name "$@" >> runs/${name}_train.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    t=$(temp)
    if [ "$t" -gt 75 ]; then
      echo "[queue] GPU ${t}C: pausing $name for 15 min $(date +%T)"
      pkill -STOP -P $pid; kill -STOP $pid
      sleep 900
      kill -CONT $pid; pkill -CONT -P $pid
      echo "[queue] resumed $name at GPU $(temp)C $(date +%T)"
    fi
    sleep 5
  done
  wait $pid; echo "[queue] end $name rc=$? $(date)"
}
sleep "${START_DELAY:-0}"
run sf_both   --resume runs/sf_both/last.pt
# sf_both_w dropped 2026-09-22: 2.4x params, +29% CPU latency (a deployment cost).
# KITTI finetune = shipped recipe (runs/kitti_mixed_fixed768, 1.156) with only the
# init changed; arch comes from the checkpoint. Overrides COMMON's sceneflow flags.
COMMON="--dataset kitti_mixed --root data/kitti2015/training --kitti2012-root data/kitti2012/training --epochs 300 --batch 4 --lr 1e-4 --ema 0.999 --aug strong --crop-h 320 --crop-w 768 --workers 8 --epoch-pause 10 --step-pause 0.1 --temp-target 68"
run kft_both --init runs/sf_both/best.pt
