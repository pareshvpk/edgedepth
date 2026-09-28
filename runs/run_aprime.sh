#!/bin/bash
# Option A': SceneFlow pretrain of the v3 head (backbone = s60 ep58, frozen all epochs),
# then KITTI finetune with the s60 recipe (60 ep, lr 5e-5, scale 1.5/0.5) for a clean compare vs s60 ep58.
cd /media/paresh/1426359D263580B2/Microchip/hailo-stereo
PY=~/.venvs/stereo/bin/python
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader; }
guard() {  # run "$@" in background with the 75 C / 15-min watchdog
  local tag=$1; shift
  while [ "$(temp)" -gt 75 ]; do sleep 900; done
  echo "[$tag] start $(date)"
  "$@" &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    t=$(temp)
    if [ "$t" -gt 75 ]; then
      echo "[$tag] GPU ${t}C: pausing 15 min $(date +%T)"
      pkill -STOP -P $pid; kill -STOP $pid; sleep 900; kill -CONT $pid; pkill -CONT -P $pid
    fi
    sleep 5
  done
  wait $pid; local rc=$?; echo "[$tag] end rc=$rc $(date)"; return $rc
}
if [ ! -f runs/sf_v3head/last.pt ] || ! grep -q "done. best" runs/sf_v3head_train.log 2>/dev/null; then
  R="--init runs/kft_both_s60/best.pt --init-partial"
  [ -f runs/sf_v3head/last.pt ] && R="--resume runs/sf_v3head/last.pt"
  guard sf_v3head $PY src/train.py --dataset sceneflow --root data/driving $R --refine v3 \
    --epochs 12 --freeze-epochs 12 --batch 8 --lr 5e-4 --crop-h 256 --crop-w 512 --workers 8 \
    --aug strong --ema 0.999 --epoch-pause 30 --step-pause 0.1 --temp-target 68 \
    --out runs/sf_v3head >> runs/sf_v3head_train.log 2>&1 || exit 1
fi
R="--init runs/sf_v3head/last.pt"
[ -f runs/kft_v3_aprime/last.pt ] && R="--resume runs/kft_v3_aprime/last.pt"
guard kft_v3_aprime $PY src/train.py --dataset kitti_mixed --root data/kitti2015/training \
  --kitti2012-root data/kitti2012/training $R --refine v3 \
  --epochs 60 --batch 4 --lr 5e-5 --scale-max 1.5 --scale-prob 0.5 --ema 0.999 --aug strong \
  --crop-h 320 --crop-w 768 --workers 8 --epoch-pause 10 --step-pause 0.1 --temp-target 68 \
  --out runs/kft_v3_aprime >> runs/kft_v3_aprime_train.log 2>&1
