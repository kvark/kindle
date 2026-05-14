#!/bin/bash
# Wrap a sweep script in a memory-limited transient systemd cgroup so
# OOM kills the job, not the whole system. The kindle joint-training
# runs can spike to 7-8GB RSS; on a 12GB box that's close enough to
# the limit that random RSS spikes have triggered system-level OOM
# kills (2026-05-14: COLDNEW killed by global OOM at init).
#
# Usage:
#   scripts/cgrun.sh /tmp/sweep_foo/run.sh
#   CGRUN_MEMORY_MAX=10G scripts/cgrun.sh /tmp/sweep_bar/run.sh
#
# Defaults: MemoryMax=10G, swap allowed up to 2G. No soft throttle
# (MemoryHigh) — soft throttle causes the process to stall in reclaim
# for hours before finally hitting MemoryMax. With just MemoryMax it
# either runs fine or OOMs immediately. Verified failure mode
# (2026-05-14 retry): MemoryHigh=7G stalled COLDNEW python for 3h
# until OOM-kill, log never wrote past init.
#
# Requires systemd user manager (default on most Linux distros).
set -u
MAX="${CGRUN_MEMORY_MAX:-10G}"
SWAP="${CGRUN_MEMORY_SWAP_MAX:-2G}"
exec systemd-run --user --scope \
    -p "MemoryMax=$MAX" \
    -p "MemorySwapMax=$SWAP" \
    bash "$@"
