# Retained Breakout policy works on AMD Auto, below the task gate

The transferred2M policy scores **67.5833** in the fixed24-natural-episode frozen
evaluation on RX7900XT/RADV, with **0/24 two-wall successes**. It completes21,976
aggregate actions/87,859 emulator frames in14.127s, plus2.609s construction,
with zero learning updates. All audits pass and the service exits successfully.
This supplies a current-host control for any separately declared warm start;
it does not demonstrate new learning or an independent learner seed.

The [bounded declaration](../experiments/2026-10-26-cdp-retained-breakout-auto.md)
uses the shared `breakout-b8/2m-frozen` export. All four source files match the
handover manifest. A separate input copy changes only saved precision BF16→Auto;
all346 tensors, optimizer/normalizer state, seed1009,12M/RGB64/B8/T16/H15/R32,
centered CDP/action-effects1 and actor_unimix0 are preserved. The native model
reports16,334,353 trainable parameters. No action aid or mixture change is used.

Evaluation seed4000001009 uses eight independent lanes, full18/sticky.25/repeat4,
native max-pooled frames and one GPU resize. First three natural episodes per
lane select24 of30 completed episodes; the other six and partial tails remain
in the evidence. The all30 log mean57.8 is not the selected24 score67.5833.
There are no truncations, action-cap cutoffs or learner updates. Full CPU replay
reproduces the saved action/outcome/frame traces and records video.

Source/input counters remain499939 learner updates and2027152 environment
actions. The frozen export advances only environment actions to2049128. All346
source/copied/frozen tensors are finite and bitwise identical; original source
hashes remain unchanged. Full config/protocol/source/counter and host-guard
checks pass. The300-second execution and600-second audit caps are unchanged.

Historical source score98.2083 used another GPU, precision and runtime. The
current lower score cannot isolate precision or hardware effects. Source
actual-initial weights were not transferred, so this is a transferred-policy
measurement without a new learning-gain claim. Fresh compact200k policies use
far less training and capacity; their scores are not matched-budget controls.

Source at launch`0b151c6`, native`bfb8c31`, runner`1ed07af`.
Native SHA256`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Artifacts: `/x/Temp/kindle-cheap-20261025/breakout-retained-2m-auto-frozen`.
Review SHA256`e691480b2f5baf22d3b5dd984f393e16a40c05f273c60776237cb4e337760ab9`.
No retry or warm-start training is part of this check. The declared Qbert
coefficient test is selected next; further Breakout learning needs a specific
changed hypothesis and must account for empty replay, games, belief and RNG.
