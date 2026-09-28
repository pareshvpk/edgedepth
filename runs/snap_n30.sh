#!/bin/bash
# Keep every epoch of kft_both_n30: copy last.pt to runs/kft_both_n30_snaps/epNN.pt
# whenever it changes, for per-epoch band evals and weight averaging with kft_both.
# Read-only on the run; exits once kft_both_n30 has finished (its log says done.).
cd "$(dirname "$0")/.."
mkdir -p runs/kft_both_n30_snaps
PY=~/.venvs/stereo/bin/python
last=""
until grep -q "^done\." runs/kft_both_n30_train.log 2>/dev/null && [ -n "$last" ] && [ "$(stat -c %Y runs/kft_both_n30/last.pt)" = "$last" ]; do
  f=runs/kft_both_n30/last.pt
  if [ -f $f ] && [ "$(stat -c %Y $f)" != "$last" ]; then
    sleep 5; m=$(stat -c %Y $f)
    ep=$(CUDA_VISIBLE_DEVICES="" $PY -c "import torch;print(torch.load('$f',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
    [ -n "$ep" ] && cp $f runs/kft_both_n30_snaps/ep$(printf %02d $ep).pt && last=$m
  fi
  sleep 15
done
