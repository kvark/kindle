# Guarded native qualification for frozen actor posterior averaging

Select one execution of the compiled fixture after the
[development checks](2026-11-06-cdp-actor-posterior-average.md) and packaged
Python suite pass. Production source is `fddf48bacc6ead2e51a7d5f31eedd6d85e045bb7`.
Native SHA256 is `960667d10fb031ec82963f5cd644802bd650c273c64db914ec6c46f6a6265680`;
test executable SHA256 is `77ab9956ba5545f1e4f770127d93d9a5447da565d334571dc4e55e3ee0ea7322`.
The build receipt pins 249 source files and the packaged artifacts.

Run only `dreamer::agent::vector::posterior_average_tests::posterior_average_native_values_state_rng_and_frozen_guards`
with `--exact --ignored --nocapture --test-threads=1`. Two Tiny RGB32 streams,
eight additional posterior draws and four synthetic decisions; zero Atari
actions, gradients or learner updates. The ordinary and averaged agents receive
identical synthetic images and forced commands. Their original recurrent state,
posterior RNG and policy RNG must remain identical at each decision. Environment
counters advance equally; learning counters, parameters and moments stay fixed.

Use a deliberately nonuniform actor. Independently reconstruct categorical
samples in F64 from captured current logits and the additional fixed draws,
evaluate the ordinary native actor on those explicit states, and average its
softmax distributions in F64. Require absolute probability error <2e-4 against
the new graph, exact selected actions for the fixed draws/masks, and observable
sample-dependent probability variation. Check all 11 shared actor parameters,
pre-observation guards, conflicting settings and both rejected learning calls.
This is an inference value/routing check, not derivative or gameplay evidence.

The existing host guard must verify the current AMD RX7900XT/RADV device, boot,
kernel and amdgpu identity. Check no live Kindle service or render-device holder
before execution; use Auto and the retained AMD environment without NVML.
Cap the child at 120s and controller at 300s. Stop on genuine faults/nonfinite
results. No driver reset/reload/reboot, retry, Atari evaluation or full learner.

Root `/x/Temp/kindle-actor-posterior-average-20261106`. Pin this protocol,
controller, queue, helpers, source and binary before execution; retain outputs,
guard receipt and terminal identity. Preserve any original failure. A successful
fixture only permits considering a separately declared frozen compatibility/cost
check; it does not automatically select that check or a trained comparison.
