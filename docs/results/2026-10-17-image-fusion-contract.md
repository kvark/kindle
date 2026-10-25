# Basic image-region fusion is already present

October17,2026. [Evidence](2026-10-17-image-fusion-contract.json) ·
[Backend plan](../backend_training_performance_plan.md).

**Missing normalization/activation fusion does not explain the remaining gap.**
Every relevant current native image region selects the shared composed path:
16 forward/backward regions in RGB, eight in CDP. It already combines bias,
optional pooling, RMS normalization and SiLU, with their adjoints. The qualified
image views additionally compose eligible conversions, packing and expansion.
This is actual program selection, not merely an available code path.

JAX also keeps convolution separate from normalization: all eight forward
cuDNN calls have `activation_mode=kNone`. Their names containing “activation”
do not establish fused SiLU. The library does include bias; native adds bias
inside its image region after the convolution's BF16 result. These are different
rounding boundaries, not a numerical-parity claim.

Keep isolated`e45b9c43`: **RGB43.197ms versus inclusive JAX27.223ms,1.59×**.
No implementation change, backend build, GPU call, learner update or game action
occurs in this audit. Production and Atari stay unchanged.

## Actual forward boundaries

An independent tuple-path reader reconciles1,215 top-level HLO nodes,3,198 read
references and17 call contexts, including all23 convolution products. Loop and
fusion bodies remain opaque execution nodes; these are not total runtime launch
counts. Source-role annotations alone are insufficient because XLA shares bodies
between compatible encoder/decoder shapes. The audit follows actual arguments
through calls, tuple extraction and bitcasts.

| Connected image role | JAX forward fusion nodes | Native core image kernels |
| --- | ---: | ---: |
| Encoder cnn0 /cnn1 |4 /4|1 /1|
| Encoder cnn2 /cnn3 |3 /2|1 /1|
| Decoder conv2 /conv1 /conv0 |3 /4 /4|1 /1 /1|
| Decoder spatial normalization |2|1|

The26 JAX nodes include associated pooling-mask preparation or layout work.
Native still has some separate layout operations, and the decoder's spatial
input addition is excluded from both columns. Thus this is **not a3.25× dispatch
advantage, matched instruction count or timing comparison**. It establishes
that another generic normalization/SiLU fusion pass would duplicate existing
native work. Fusing an entire convolution with normalization is a different
ownership/resource question, not something JAX's current calls already do.

All eight JAX regions contain a BF16 normalization-rounding boundary before
the activation result, with further reduced-precision activation operations.
Native's image arithmetic and saved pooled values stay F32, with explicit output
conversion. Both use F32 norm scales. Matching the JAX body by moving/removing
rounding would be a precision/semantic change, not a free composition.

## Backward values cannot simply disappear

Every native forward/backward pair is matched by its actual scale parameter.
The backward kernel reads the forward kernel's saved pooled values and inverse
RMS, plus the winner index for pooled regions. All15 RGB and eight CDP scale/
bias reduction chains reach the exact declared parameter-gradient buffers:
30 and16 `SumRows` dispatches respectively.

Declared pooled/inverse/winner buffers total313,851,904bytes RGB and
208,306,176bytes CDP. These are logical tape extents, **not traffic, peak VRAM,
latency or removable savings**. They exclude activation outputs reused by
convolution derivatives. JAX retains/recomputes different boundary values;
its cross-adjoint value census is preserved raw, not compared as an equivalent
memory total. Dropping native tapes without recomputation or another exact
representation would change backward behavior.

## Verification, failures and decision

Four shape fixtures, four malformed-shape rejections and eight independent
wrong-version/port/input/operation/role/tuple-path rejection tests pass. The
original review's “six negative checks” included two simple equality sanity
checks; the publication audit distinguishes them and adds actual refusal tests.
All29 source/result seals pass. Final CPU review finishes**02:28:37UTC**;
all workers are terminal. Upstream remains Meganeura41a15d31/Blade4845d652.

Preserve the first reader's flat-tuple failure at a nested recurrent-state
extraction. Recursive shape handling then encounters XLA's tuple-index display
comments; a separate lexical repair accepts those comments without dropping
structure/leaf checks. The local exploratory `inspect.py` also shadows Python's
standard module during traceback reporting; exclusive output creation prevents
rewriting the retained inventory. Later readers exclude that directory from
module search. Original sources/logs remain. No GPU retry or raised limits.

Do not start another image-fusion, precision, layout or geometry search from
these counts. The next task is a **CPU contract for boundary-matched timing** of
current world/imagination work: first reconcile retained timestamp coverage,
then determine whether in-pass markers can preserve cached commands, barriers,
submissions and query lifetimes. Existing per-dispatch profiling changes pass
boundaries; earlier coarse chunking adds waits. Neither may silently substitute
for ordinary execution. Any new observer needs an explicit overhead/numerical
gate before a separate bounded collection; no GPU allocation or backend edit
is implied here. Library feeding, recurrence and the full parity goal remain
open, while the failed UBO/normalization queues stay closed.

Artifacts: `runs/image-fusion-contract-20261017.y8KNZDkG`.
