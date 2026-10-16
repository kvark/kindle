# Review retained matrix counters independently

October16,2026. Both GPU cohorts in the [matrix diagnostic](2026-10-16-matrix-feeding-counters.md)
have finished:65 successful captures, plus the retained original observer timeout.
The first complete CPU reviewer stops on tiled-000 because the independently
collected shared-bank-conflict total is147,602, while read+write is147,589.
Across all65 records,44 fail that exact identity; the largest relative difference
is.12737144%. Preserve the original reviewer, log, failed unit and controller.

Do not change the numerical oracle, accept a new counter tolerance, alter raw
counter images, or rerun a GPU job. A separate CPU review reports the three
counters independently and records every signed difference and the failed exact
identity. Offline metric reevaluation must still match captured JSON exactly;
all original output/operand/guard/source/pass/launch checks stay exact. Use a
new offline-output filename, retaining the first review's decoded output.

The SDK exposes separate raw dependencies for read and write metrics. NVIDIA
documents [multi-pass counter variability](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-and-precision);
this is a plausible explanation, not proof of this discrepancy's source or a
complete account of other shared operation classes. Do not reconstruct a total
from parts, clamp it, or identify a shader site from these counters.

One CPU,2GiB,zero swap,720s, no restart. No backend build, GPU call, production
change, Atari restart or automatic promotion. Artifact root remains
`runs/matrix-feeding-counters-20261016.dKgyoaOO`.
