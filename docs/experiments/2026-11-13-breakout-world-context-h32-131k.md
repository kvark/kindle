# 4M context-frame world with a 32-step rollout for 131,072 updates

The 4M context-frame world under the 32-step rollout improved from 32,768 to
65,536 updates at every horizon on the held-out corpus (thirty-two steps 28 to
37 of 116, sixteen steps 42 to 51 of 111) and its development curve was still
rising at the end, while a 12M model at 32k showed that budget buys more than
capacity in this range. This run doubles the budget once more: a fresh life
of `context-config-96-4m-h32.json` with `--steps 131072 --eval-every 8192
--dense 128 --prediction-length 32 --pixel-loss motion_event_rollout` on
`corpus/fit96` and `corpus/dev96`. Deterministic: its first eight points
reproduce the 65k run; new information starts at 73,728 updates. Output
`train-96-context-4m-h32-131k`, unit `kindle-world-context-h32-131k-20261113.service`,
direct launch under a 24G memory cap without the host guard; about 32 hours
at 867 ms per update. The final checkpoint is scored on the held-out seed
777001 corpus with `probe`. Read against the 65k point (development
112/99/75/61/32, held-out 110/90/72/51/37); a continued rise at sixteen and
thirty-two steps says the horizon is still budget-limited, a plateau says the
recipe has reached its ceiling at this size. One run; development screen plus
one held-out corpus.
