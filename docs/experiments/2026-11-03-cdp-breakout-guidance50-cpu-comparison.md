# CPU-only correction of the guidance comparison

Source inspection during the original learner, before its frozen outcomes,
finds a redundant equality assertion in `full/compare.py`: the loop includes
`exploration`, although the preceding complete-header check correctly requires
the declared probability change .20 to .50. The original headers therefore
cannot pass this equality. The helper qualification tested the complete-header
validator and comparison arithmetic separately, missing this combined path.

Keep the original 117 pinned inputs, learner, four frozen arms, replays, audits,
curve and terminal collector unchanged. Preserve the original service outcome
and journal. After all GPU stages finish and both original main and collector
are terminal, run one separate CPU correction under
`/x/Temp/kindle-breakout-guidance50-20261103/cpu-comparison-repair`.

The corrected copy removes only `exploration` from the redundant equality loop;
the original complete-header validator still requires exactly .20 to .50,
including unchanged kind, seed, cap and all other non-metadata fields. Exercise
the actual original and corrected comparison entry points with synthetic final
results: reproduce the original rejection, accept the declared header pair and
reject wrong probability, cap, teacher identity and learner settings. Do not
construct a native agent or emulator, or read candidate frozen outcomes during
qualification. Pin original and corrected sources and qualification evidence.

Run the corrected comparison once against saved evidence only after the original
failure is preserved. Write its result and closure exclusively into the repair
directory. Require the original failed assertion to be `exploration`, all five
GPU-stage receipts and guards to pass, the complete learning review, four frozen
replays, curve and exposure results to exist, and the original recovery receipt
to pass. Verify all original and retained pins and capture input/output hashes.
Check that those artifacts remain unchanged after the correction. Do not rerun
training, frozen assessment, replay, plotting, successful audit or collector.

Keep the original primary greedy comparison, lane bootstrap, time budget,
actual-initial controls and 864/90%-two-wall gate unchanged. This repairs an
analysis assertion, not a protocol or outcome. If successful, report
`complete_with_cpu_audit_repair`; the original services remain failed and no
original passing completion is fabricated. Record CPU co-host qualification
time without subtracting it from training. No successor learner is selected.
