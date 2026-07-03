#!/bin/bash
# Compact report on the queued experiments.
# Run after seq.log shows "ALL DONE" markers.
set -u
LOGDIR=${LOGDIR:-/tmp/aff_runs}
ANALYZE="$(dirname "$0")/analyze_arc_runs.py"

if [ ! -f "$ANALYZE" ]; then
    echo "error: analyzer not found at $ANALYZE" >&2
    exit 1
fi

echo "=== sequence log ==="
cat "$LOGDIR/seq.log" 2>/dev/null || echo "(no $LOGDIR/seq.log)"

echo
echo "=== single-pair (tu93+sp80) ==="
python3 "$ANALYZE" \
    "$LOGDIR/base.log" \
    "$LOGDIR/change05.log" \
    "$LOGDIR/xgame.log" \
    "$LOGDIR/combo.log" \
    "$LOGDIR/centroid.log" \
    "$LOGDIR/all_feat.log"

echo
echo "=== cold-start (5 unseen games) ==="
python3 "$ANALYZE" \
    "$LOGDIR/coldnew_base.log" \
    "$LOGDIR/coldnew_full.log" \
    "$LOGDIR/all_feat_coldnew.log"

echo
echo "=== pretrain → resume transfer ==="
python3 "$ANALYZE" \
    "$LOGDIR/pretrain20.log" \
    "$LOGDIR/resume5.log"
