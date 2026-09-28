#!/bin/bash
# Copy runs/kft_both/best.pt to runs/kft_both_snaps/ep<N>.pt whenever best.json's epoch
# changes, so stage 2 can init from an earlier checkpoint (0-5 m drifts late).
# Read-only on the run; exits when screen_queue.sh does.
cd "$(dirname "$0")/.."
last=""
while pgrep -f '^/bin/bash runs/(screen|resume)_queue.sh' >/dev/null; do
  ep=$(python3 -c "import json;print(json.load(open('runs/kft_both/best.json'))['epoch'])" 2>/dev/null)
  if [ -n "$ep" ] && [ "$ep" != "$last" ]; then
    sleep 3  # best.json is written after best.pt; let the write settle
    cp runs/kft_both/best.pt runs/kft_both_snaps/ep$(printf %03d $ep).pt && last=$ep
  fi
  sleep 20
done
