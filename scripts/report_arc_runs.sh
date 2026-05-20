#!/bin/bash
# Compact report on the queued experiments.
# Run after seq.log shows "ALL DONE" markers.
set -u
LOGDIR=/tmp/aff_runs

echo "=== sequence log ==="
cat $LOGDIR/seq.log 2>/dev/null

echo
echo "=== single-pair (tu93+sp80) ==="
python3 $LOGDIR/analyze.py \
    $LOGDIR/base.log \
    $LOGDIR/change05.log \
    $LOGDIR/xgame.log \
    $LOGDIR/combo.log \
    $LOGDIR/centroid.log \
    $LOGDIR/all_feat.log 2>/dev/null

echo
echo "=== cold-start (5 unseen games) ==="
python3 $LOGDIR/analyze.py \
    $LOGDIR/coldnew_base.log \
    $LOGDIR/coldnew_full.log \
    $LOGDIR/all_feat_coldnew.log 2>/dev/null

echo
echo "=== pretrain → resume transfer ==="
python3 $LOGDIR/analyze.py \
    $LOGDIR/pretrain20.log \
    $LOGDIR/resume5.log 2>/dev/null
