# Same-backend block-matmul runtime qualification

Status: **declared and waiting; 103 CPU checks pass, no GPU result**. Finish
Pong root 1009's entire original pair, then both latest-dependency stages, then
this optimization comparison. Remaining Pong roots stay held. No learning,
adoption or five-game completion follows automatically.

## Fixed inputs and live handoff

The [declaration](../../runs/block-matmul-conv-runtime-20260913.fJSirl/declaration.md)
and [manifest](../../runs/block-matmul-conv-runtime-20260913.fJSirl/manifest.json)
bind **43,151 pins**. Manifest SHA-256:
`f94d6bd500232601e848b58f8f4628892e7a7a02210ca90704ba6f96e21c55fa`.
Compare **58f328a / native fa6bdd2a** against **20b9b8a / native 5e4ea9e1**;
both use **Meganeura 75dfe901 / Blade f6f2729e**. The direct upstream recheck
still finds 75dfe901. The source-matched [packages and six release fixtures](2026-09-10-block-matmul.md#latest-source-package-and-release-fixtures)
are preserved, not rebuilt or adopted by this declaration.

Follower **282009/start ticks 15020597** launches at **08:04:19 UTC** on
September 13, bound only to full-runtime follower **273381/14615337**. The
[08:06 handoff audit](../../runs/block-matmul-conv-runtime-20260913.fJSirl/handoff-audit.json)
reverifies every pin, actual declaration/launch receipts and all 103 CPU results.
All three waiters and all four original Pong processes retain their identities;
no waiter has a native child or GPU-stage output. Both real entrypoints refuse
live predecessors before new GPU work. These are scheduling/CPU checks, not
native qualification. Do not edit the live inputs or launch a competing worker.

The worker requires its bound predecessor's actual exit, full raw nineteen-test
hardware proof, six canary/ten pixel windows, complete state/optimizer/trace
checks and successful command/follower history. It retains the complete original
Pong/Freeway evidence and exact requested no-spawn boundary. Valid competence
failures remain failures; missing evidence, another exit or changed inputs,
upstream or host stop execution without retries.

## Actual adapter selection

Default test sessions are environment-independent. Both new comparison arms
therefore set `VK_DRIVER_FILES` to the pinned NVIDIA driver manifest; this
restricts driver discovery without changing host configuration. The selector is
documented by [Khronos](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md#overriding-the-default-driver-discovery).
It enables no Meganeura tuning, tracing, initialization or queue option.

The gate pins the installed loader, driver library, manifest, library resolution
and `vulkaninfo`. Before tests, restricted enumeration must report exactly one
RTX 5080, vendor/device **0x10de/0x2c02**, discrete NVIDIA proprietary driver
**595.91.07**, with a nonzero device UUID. Preserve raw stdout/stderr and verify
the same host and loader resolution before native work. Reject undeclared loader
overrides or elevated execution. The parser's positive/negative fixtures and
CPU file checks pass; **actual enumeration has not run**. NVIDIA memory polling
alone is not proof of the executing adapter. No wrong-adapter execution is claimed.

## Acceptance remains unchanged

- Run both block GPU tests first: upstream F64 composed losses/all gradients,
  then Kindle's production exact loss/output/all-gradient comparison. Follow
  with the nineteen retained full-world/cache/reset/vector/restore tests.
  All **21 exact named tests** must pass with their pinned binaries.
- Run **six canaries**: parent/candidate at update 1, parent/candidate at
  update 8, then candidate/parent at update 8. Require exact **241 tensor entries
  and 146 optimizer moments**, metadata/normalizers and non-timing reports.
  Fresh parents must reproduce the newly qualified 75dfe anchors; historical
  ce80 CPU fixtures are not reinterpreted as new-backend results.
- Run unchanged **N6/R256/B16/T64/full-BPTT/F32/12M** Boxing pixel **AB/BA**.
  Each arm gets fresh seed 7301, **3,840 actions / 610 updates**, then a sampled
  frozen seed-8301 restore of **768 actions / zero updates**. Full state,
  headers, action/reward/episode/reset traces and learner reports must match
  exactly, including the retained 75dfe parent. Keep the candidate's unchanged
  Freeway .25/hold64 override integration and unassisted frozen check.
- Time only the warmed **2,304–3,840-action** window: 1,536 actual actions,
  384 updates, bounded debt and actual emulator-frame clocks. Both ratios
  **≥0.98** pass regression acceptance; optimization qualification separately
  requires **both >1.005**. Neither establishes super-real-time learning or
  seed reliability.
- Serialize all work. Cover the hardware group, six canary and ten pixel
  windows with 250 ms direct free/reserved/used-memory samples, gaps no larger
  than 1.5 seconds and **at least 2,048 MiB directly free**. Recheck raw commands,
  complete state, traces, adapter evidence and memory before reporting a result.

No game roots, budgets, paired controls or competence thresholds change here.
Boxing remains the only confirmed three-root game. Any later learning use needs
the [new matched declaration required by the throughput decision](2026-09-13-throughput-priority.md).
