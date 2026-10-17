# One T16/T64 production preflight before more learning

The five-game quality goal is unfinished. Latest upstream qualifies but does
not improve the roughly108ms 12M update. Test one existing configuration change:
longer training sequences, not another model or representation sweep.

Two fresh Breakout runs, seed103, **4,096 actual actions each**: centered CDP,
12M/N8/B8/R32/H15, with T16 versus T64 and matching full BPTT. Everything else
is the qualified centered-smoke recipe: exploration coefficient1, microbatch8,
replay100000, split rates/cosine500/AGC.3, native frames/one GPU RGB64 resize,
full18/sticky.25/repeat4/no reset no-ops. Native7311547d, Meganeura31026833,
Blade56f0565. No source rebuild or learning-code change.

After each run, freeze its final model for1,024 actions, saving another
checkpoint. Total **8,192 training +2,048 frozen actions**, plus no synthetic
or offline learner updates. Save actual initial weights and require identical
initial346 tensors between arms. Frozen weights/optimizer tensors must be exact;
all metrics/checkpoints finite, expected native device and >=2GiB sampled Vulkan
estimated headroom throughout. Audit all per-stream transitions, resets, replay
and training credit with the independent configuration-aware vector auditor.
Do not assume one update per four actions or force fractional final debt to zero.
These are short correctness/resource smokes, not competence evaluations or
independent learner replicates. Keep every episode and unfinished tail.

Compare the fixed last2,048 actions: all512 T16 updates versus128 T64 updates,
**65,536 replay positions each**, using ordinary full-update timing reports.
Report every stage, ms/update, ms/replay position, sampled headroom and whole-run
actions/time. No dispatch timestamp instrumentation, concurrent heavy work or
compilation. Retain all warmup/initialization/checkpoint costs separately from
the steady window. One measurement per arm; no tuning sweep.

T64 provides longer temporal gradients and512 rather than128 imagination starts.
The existing large-batch RSSM implementation remains unchanged; do not extend
the grouped-kernel threshold alongside sequence length. Backend qualification
is scoped evidence, not proof of arbitrary new-shape F64 parity. This preflight
checks production accounting, finite learning and restore integrity; it does
not compare gradients for two mathematically different BPTT objectives.

Changing T also changes centering statistics, warmup experience and optimizer
updates per action. This is **not an unchanged-learning speedup**. Consider a
separately declared matched three-seed learning comparison only if validity and
memory pass and steady cost per replay position falls by at least10%. Otherwise
review the bottleneck; do not relax the gate or automatically extend training.

Four serial host-guarded jobs in one30min persistent service, training workers
600s each and frozen workers300s; Restart=no, KillMode=control-group. Ordinary
GPU-worker CPU allocation, RTX5080/580.178.04, same declared boot. Allocation
warnings are retained/nonblocking; API/numerical/hard faults and deadlines stop
for review. No blind retry, NVML polling or recovery. CPU preparation/audit:
one CPU,2GiB,zero swap. Source/native/job hashes sealed before launch.

Artifacts: `runs/cdp-sequence-preflight-20261009.EBJhphnr`. No existing learner
is resumed or repinned. All five quality targets and the RGB budget question
remain open after this preflight, regardless of its timing result.
