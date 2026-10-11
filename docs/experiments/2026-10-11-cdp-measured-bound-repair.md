# Correct the logical-memory preflight, without changing numerical gates

The [original trial](2026-10-11-cdp-measured-build.md) is closed as a failed
preflight, not an optimization result. Its search returns "no qualified program
within the search bounds" in9.94s, before any application qualification or
timing. Zero optimizer updates/game actions; exact346 calibration tensors;
no new allocation warning or recovery. The original source, binary, declaration,
guard failure and [CPU review](../../runs/cdp-measured-build-20261011.4TQhs8Cc/failure-review.json)
remain. No numerical candidate was rejected.

The chosen4GiB bound counts **logical buffers**, not actual allocated VRAM.
The retained ordinary imagination profile already totals9,788,951,108 logical
bytes but2,400,187,972 allocated bytes. The upstream selector sums logical
buffers for incumbent plus challenger, adding any persistent-write snapshots.
The failed API returns no per-program report; its exact new-plan sizes are
unavailable. This is an application preflight error, not evidence against
measured construction or a GPU fault.

After that review, declare exactly one corrected trial:

- Change only the logical pair-plan cap4→20GiB. Record the current ordinary
  logical-buffer size before search. Do not change the2GiB sampled Vulkan
  estimated-headroom floor, graph, precision, tolerances, inputs or weights.
- Keep Meganeura2c42b448/Blade56f0565/native0f26ee63. Its19-job qualification
  and ordinary old/new timing pair have passed. Only the test binary changes;
  no production graph/native rebuild or repeated qualification campaign.
- Use the same checkpoint/fixture and latest ordinary control already measured
  at108.302ms. Preserve its exact binary and outputs. Explicitly disclose the
  test-only cap/logging change; this is not an exact same-binary pair.
- Retain all original search limits:4 graph forms/8 programs/180s soft search,
  at most4 submission chunks,16 private kernel classes/64MiB scratch, six
  sample pairs,5% whole-update improvement gate. All application-output and
  held-out first-target tolerances, exact categorical checks and sixteen warmup
  plus256 measured full updates stay unchanged. No memo reuse or F16 storage.
- One guarded600s process in a900s persistent service, no CPU quota, no builds
  or heavy analysis during timing. CPU build/review: one CPU,2GiB,zero swap.
  Allocation warnings remain recorded; faults/mismatches/deadlines stop for
  review. No retry, further cap expansion or optimization sweep follows.

Report this additional attempt and its preparation cost even if it succeeds.
A passing synthetic result is not deployed throughput or learning evidence.
No new learner allocation follows automatically; all five long-run quality
targets and the matched RGB budget question remain open.

Artifacts: `runs/cdp-measured-bound-repair-20261011.E12THDy8`.
