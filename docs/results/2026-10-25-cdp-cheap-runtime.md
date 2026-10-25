# Smaller native CDP recipes

The user permits game-specific capacity and resolution to reduce training cost.
Size4M adds an intermediate preset (deter1024, hidden128,32 stochastic variables
with8 classes,8 blocks, units128, vision depth8). Existing presets are unchanged.
Native RGB32/RGB48/RGB64 now select preprocessing, replay and CNN geometry.
`--image-size` chooses fresh geometry; restore reads checkpoint geometry and
rejects overrides. The default remains64, with one GPU resize from native frames.

Source: preset`5af5653`, native resize`bfb8c31`, Atari CLI`a99416e`.
Meganeura/Blade pins, precision policy and GPU driver are unchanged. Native SHA256
`f2606ac28f5f293b73ccdb3f408278c6c584e50ccc73d52fd605a39b231609a5`.
Build/wheels, runtime, test receipts and visual references live under
`/x/Temp/kindle-cheap-20261025`.

Validation:123 Rust CPU tests pass,78 GPU tests ignored by that suite; strict
Rust Clippy and427 relevant Python tests pass. New32/48 fixtures match independent
Pillow12.3 bilinear bytes, preserving all64 fixtures. A guarded GPU reference test
compares32/48/64 output against scalar results for seven source shapes, through
CPU-upload and resident-GPU inputs. It passes on RX7900XT/RADV.

| Bounded smoke | Trainable parameters | Actions / emulator frames / updates | Training wall | Actions/s |
|---|---:|---:|---:|---:|
| Boxing1M/RGB32 |834673|4096 /16384 /482|7.085s|578.11|
| Pong4M/RGB48 |3578225|4096 /16384 /482|11.724s|349.37|

Both use centered CDP/action-effects1, Auto, N8/B8/T16/H15/R16, microbatch8,
native max-pooled frames, full18/sticky.25/repeat4 and seed1009. Updates start
at248; zero debt. Each1024-action frozen restore has zero updates and preserves
all346 tensors. The prior64x64 Freeway checkpoint also restores for1024 actions
with all346 tensors unchanged. Finite/config/counter checks and all six host
guards pass; the validation service exits successfully.

These are bounded execution checks, not competence tests or an isolated
resolution speedup. Reducing geometry also reduces flattened encoder embeddings
and CDP head dimensions; recurrent/imagination work remains. CPU-only visibility
references sample four frames in each of the five games under fixed random
actions. Both Boxing fighters remain visible at32. Pong's ball has more contrast
at48, while Breakout's tiny ball is substantially attenuated at32; keep Breakout
at64 initially. Frozen learning outcomes must determine whether smaller inputs
are useful. The next declared allocation is Boxing1M/RGB32 at200k actions.
