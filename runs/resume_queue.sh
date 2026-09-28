#!/bin/bash
# Resume the EPE push after the 2026-09-22 shutdown (kft_both stopped after epoch 176).
#   nohup runs/resume_queue.sh >> runs/screen_queue.log 2>&1 &
# 1. kft_both --resume runs/kft_both/last.pt (restores opt/sched/EMA) to epoch 299
# 2. stage 2: kft_both_n30 then kft_both_s60 (runs/stage2_queue.sh)
# Helpers: 1 Hz GPU log, snap_best.sh (every new kft_both best), snap_n30.sh (every n30 epoch).
# Thermal: --temp-target 68 governor + watchdog (>75 C -> SIGSTOP 15 min), user rule.
cd "$(dirname "$0")/.."
PY=~/.venvs/stereo/bin/python
COMMON="--dataset kitti_mixed --root data/kitti2015/training --kitti2012-root data/kitti2012/training --epochs 300 --batch 4 --lr 1e-4 --ema 0.999 --aug strong --crop-h 320 --crop-w 768 --workers 8 --epoch-pause 10 --step-pause 0.1 --temp-target 68"
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
nohup nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,power.draw --format=csv -l 1 >> runs/gpu_temp.csv 2>/dev/null &
nohup runs/snap_best.sh > /dev/null 2>&1 &
nohup runs/snap_n30.sh >> runs/snap_n30.log 2>&1 &
nohup runs/stage2_queue.sh >> runs/screen_queue.log 2>&1 &   # waits for this script to exit
while [ "$(temp)" -gt 75 ]; do echo "[queue] GPU $(temp)C, cooling 15 min before kft_both"; sleep 900; done
echo "[queue] resume kft_both $(date)"
$PY src/train.py $COMMON --out runs/kft_both --resume runs/kft_both/last.pt >> runs/kft_both_train.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  t=$(temp)
  if [ "$t" -gt 75 ]; then
    echo "[queue] GPU ${t}C: pausing kft_both for 15 min $(date +%T)"
    pkill -STOP -P $pid; kill -STOP $pid; sleep 900; kill -CONT $pid; pkill -CONT -P $pid
    echo "[queue] resumed kft_both at GPU $(temp)C $(date +%T)"
  fi
  sleep 5
done
wait $pid; echo "[queue] end kft_both rc=$? $(date)"
