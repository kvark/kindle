# Attribute the remainder after connected image training

The qualified image-region candidate e1a2407c changes complete RGB12M/B32
65.609→63.640ms,3.0015%. It barely crosses the fixed3% gate; no timing repeat
or robust-speed/learning claim. Before another optimization, measure whether
its combined image kernels or the remaining graph account for the cost.
Artifacts: `runs/image-remainder-20261014.bbwMyVKw`.

Use the unchanged image-region backend and model. Only an ignored diagnostic
test changes. One synthetic update initializes the exact retained fixture;
then disable optimizer/clipping/gradient accumulation. On the same complete
world graph, always retain heavy multiplication and classify the rest into:

- Image-region forward/backward kernels and their parameter-gradient SumRows
  tails, traced from the backward partial outputs.
- All other non-heavy operations.

Run all four enabled/disabled combinations, three counterbalanced repetitions,
two warmup+16 measured replays per condition, retaining dependencies/bindings
and cached command execution. Full before/after baselines and exact output/
292-tensor restoration checks must pass. Preserve metadata for every dispatch,
all raw samples and failed checks. These interacting fixed-graph allocations
are not pure kernel latency, ordinary complete-update timing or additive savings.
One five-minute guarded process, one synthetic update,367 optimizer-free
world replays, zero actions. No GPU correctness retry or optimization sweep.

Require identical five full plans/pipelines, actual initial/first tensors,
first outputs and first metric report versus the just-completed candidate.
The attribution changes one-chunk replay boundaries, not ordinary four-chunk
performance. Numerical APIs/faults/deadlines stop for review; standalone
allocation warnings are recorded. Existing device/headroom/host-only monitoring
rules apply. Persistent systemd service, Restart=no, KillMode=control-group,
bounded deadline, no GPU CPU quota. CPU preparation/review uses1CPU/2GiB/0swap.
No new production pin, Atari work, broad capture or whole-goal closure.
