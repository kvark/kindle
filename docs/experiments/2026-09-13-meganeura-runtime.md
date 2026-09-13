# Latest-backend full-state and pixel qualification

Status: **declared and waiting; CPU checks pass, no GPU result**. This is the
separate continuation after the [75dfe901 hardware stage](2026-09-13-meganeura-conv.md).
It preserves the user's decision: finish Pong root 1009's entire active pair,
then qualify throughput before the held roots. It changes no active experiment
input and starts no subsequent learning or adoption automatically.

## Bound handoff

The [declaration](../../runs/meganeura-conv-learning-20260913.461N2c/declaration.md)
and [manifest](../../runs/meganeura-conv-learning-20260913.461N2c/manifest.json)
bind **31,711 pins and 55 passing CPU tests**. Manifest SHA-256:
`04a8ba758f3b1b755f4a1c7fd5ff7ef6cbc14855da50ad5e63bf44c508461606`.
The real entrypoint refuses the still-live original pair before native-bearing
imports, output creation or GPU queries. Tests reject changed recipes or
identities, incomplete/mismatched state, traces, command histories and hosts.
The qualified ce80 control's complete raw evidence is independently reread
without invoking its completed writer.

Follower **273381/start ticks 14615337** launches at **06:56:46 UTC** on
September 13, bound only to first-hardware follower **269283/14332226**.
The [07:00 live handoff audit](../../runs/meganeura-conv-learning-20260913.461N2c/handoff-audit.json)
verifies every pin, actual declaration/launch receipts and both waiting process
identities. The original scheduler, Pong controller, trainer and logger are
unchanged. Neither GPU stage has a child or native output. Preserve these live
scripts and declarations; do not launch a competing worker.

After the bound predecessor exits, the worker must independently verify its
nineteen raw hardware results, whole-window direct-memory coverage and successful
command/follower history. It also retains the complete original Pong/Freeway
evidence and exact requested no-spawn boundary before root 2017. A valid
competence failure is retained; incomplete data, a different exit, changed
upstream or a runtime failure stops execution without retries.

## Fixed comparison

Compare qualified Kindle **1e00e818 / ce80e9cd / registry Blade 0.9.0 / abf4ae5d**
against dependency-only **58f328a / 75dfe901 / shared Blade f6f2729e / fa6bdd2a**.
Use each isolated package's matching runner and auditor. Verify the actual
LeVJEPA encoder and recovered **595.91.07** driver. No block-matmul, autotuning,
tracing, initialization, queue or learning-setting change is enabled.

- Six prediction-only 12M/B16/T64 full-recurrence canaries: parent/candidate at
  update 1, parent/candidate at update 8, then candidate/parent at update 8.
  Require exact complete state, all **241 tensor entries and 146 optimizer
  moments**, normalizers and non-timing reports. Fresh controls must also match
  the retained ce80 update-1 and update-8 anchors. Zero initial learning rate
  does not exempt optimizer moments.
- N6/R256/B16/T64/full-BPTT/F32 Boxing pixel **AB/BA**: four fresh seed-7301
  arms, each **3,840 actions / 610 updates**, followed by a sampled frozen
  seed-8301 restore of **768 actions / zero updates**. Require exact full-state,
  header, action, reward, episode, reset and learner-report signatures across
  all arms and the retained same-driver ce80 pixel control.
- Time only the warmed **2,304–3,840-action** window: 1,536 actual actions and
  384 updates, bounded debt and actual emulator-frame clocks. Both candidate/
  control ratios must be **at least 0.98** for regression acceptance; a speedup
  claim separately requires **both above 1.005**. Neither means super-real-time
  learning or improved seed reliability.
- Retain the candidate's Freeway **.25/hold64** override integration at the
  same 3,840/768 budgets, with strictly unassisted frozen evaluation. This is
  an interface check, not a learned Freeway result.

All **six canary and ten pixel native windows** require serialized GPU work,
same-host/idle-device guards and directly logged free/reserved/used memory at
250 ms. Reject gaps over 1.5 seconds or directly free memory below **2,048 MiB**.
The independent final checker rereads raw commands, complete checkpoints,
reports, traces and coverage before qualification; summary JSON alone is
insufficient. The retained-control and negative CPU fixtures are not new GPU
evidence.

After this dependency gate, the [same-backend block comparison](2026-09-10-block-matmul.md#september-13-identical-carry-onto-latest-runtime)
still needs its own component/state/trace/memory and untraced throughput gates.
Any learning continuation needs a new matched declaration. Boxing remains the
only confirmed three-root game; queue preparation changes no competence result.
