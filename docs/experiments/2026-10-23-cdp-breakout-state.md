# Breakout: frozen spatial-state diagnosis before another training change

The 12M CDP source scores98.21 after2M actions; its warm-start continuation
scores85.17 after8M total exposure. More budget did not solve this lineage.
Greedy evaluation is inconsistent: source53.46, final97.79, actual-initial2.21.
Keep sampled evaluation as standard. Backend optimization remains parked.

## Question and bounded comparison

Can the actual current encoder and recurrent state expose ball/paddle position
on held-out trajectories, and does longer training change that ability? Earlier
1M probes do not answer this for the current12M model.

Compare unchanged source2M, final8M and actual-initial checkpoints. Collect the
same eight fixed random-action trajectories,4096 actions each, per model using
the existing published full18-action/sticky.25/native-input protocol. RAM labels
are diagnostic only: they never enter actor inputs, rewards or actor training.
Use the existing train/validation/test seed splits and five native GPU MLP heads:
pixels, CNN, posterior, h1 prior and h15 prior. Each head gets2048 updates,
hidden128/batch64, validation-only selection; no test-set fitting or retries.
Use training-only normalization and retain constant, unrelated-action and
persistence controls. Actor parameters stay frozen throughout.

The checkpoint defines actual widths: deterministic2048, CNN1024 and latent2560.
The h1 deterministic check covers all2048 coordinates at the unchanged2e-5
tolerance. Verify346 frozen tensors, trace identities, independent readout
metrics, guards and source receipts. Total98,304 diagnostic actions and30,720
readout updates; zero actor updates. Per-model timeout900s, total service4200s.
GPU worker12GiB/no swap; CPU audit2GiB/one CPU. No backend build or timing sweep.

The384-action/160-readout-update smoke passes GPU capture and frozen checks.
Its first CPU auditor imports the historical helper and fails; a corrected
import-order CPU invocation passes without repeating capture. Preserve both.
Full controller `study_v2.py` fixes only import priority/source self-identification.
Artifacts: `runs/cdp-breakout-state-20261023.PLPVSIid`.

## Decision

Good held-out spatial readouts move the next training experiment toward policy,
reward/exploration or imagination rather than replacing the encoder. A consistent
loss between CNN and recurrent predictions supports a representation/dynamics
training experiment. Failed readouts alone cannot prove absent information;
especially a weak pixel control limits interpretation. Do not expand this into
a probe architecture search. All five Atari competence targets remain required.
