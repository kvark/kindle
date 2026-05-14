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
# Defaults: MemoryMax=8G, MemoryHigh=7G, no swap. Leaves ~4GB
# headroom for the system on a 12GB box. Adjust via env vars.
#
# Requires systemd user manager (default on most Linux distros).
set -u
MAX="${CGRUN_MEMORY_MAX:-8G}"
HIGH="${CGRUN_MEMORY_HIGH:-7G}"
exec systemd-run --user --scope \
    -p "MemoryMax=$MAX" \
    -p "MemoryHigh=$HIGH" \
    -p "MemorySwapMax=0" \
    bash "$@"
