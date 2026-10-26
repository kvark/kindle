# Pong learning-rate candidate after the B16 tradeoff

**Selected after completed Qbert evaluation and analysis.** Qbert12M48/B16
scores959.375/0of24 pyramids at63.70% more training time than4M/B8, with lower
cube coverage. No Qbert extension is selected. The exact staged CLI patch is
applied only after its training, frozen arms, audits and comparison finish.
All278 applied vector tests pass; code commit`22bd61b`, native unchanged.

PongB16 uses22.37% less training time at500k actions but scores11.875 versus B8's
14.958, both24/24 wins and below mean15. B16 takes62,454 optimizer steps versus
124,939, with nearly equal sampled replay positions. This motivates a learning-
rate candidate; it does not prove that fewer steps caused the lower score.

The existing native configuration supports base, encoder and dynamics rates.
`configure_d3_optimizer` applies encoder/base and dynamics/base multipliers;
behavior defaults to the base rate. The prior CLI fixed CDP at4e-5/6e-6/4e-4.
The applied change permits the existing `--learning-rate` on fresh CDP runs and
scales the two split rates proportionally. Default values are unchanged.
The candidate8e-5 base implies1.2e-5 encoder and8e-4 dynamics, with behavior using
the base. The1,000-update warmup, optimizer moments, clipping, slow critic and
return normalization schedules remain unchanged. This is a testable rate choice,
not an equivalence to B8 or a general batch-scaling rule.

The staged CLI validates finite positive F32 rates, including derived split-rate
overflow/underflow, before output or environment construction. Fresh non-CDP
split defaults remain unchanged; explicit restore overrides remain forbidden.
All278 staged vector tests pass, including the existing default CDP recipes and
new half/default/double-rate forwarding. No native rebuild, GPU construction,
training or production source mutation occurs in this preparation.

Staging root:`/x/Temp/kindle-cheap-20261025/cdp-learning-rate-cli`, with
`python/examples/atari_vector.py`, `python/tests/test_vector.py`, `change.patch`,
`receipt.json`, `validation.xml` and `validation.json`. Patch SHA256
`209440271ae5a7642ed311897b411478cb62324004b4996116f7e13dea9f353a`;
staged runner`1e4b7c83768c82226200cb41b166543a75ab5993b750593cc3e88a44f3be54da`;
JUnit`e898f69c5d713e591713423f51a13745f0fc1878e381690085dc34f6a792169f`.
The previous runner was`8b8491e75acac43eedfa48ce53b1bd179e91c9c2277d02169628bd4e7c470e9b`.
The applied runner matches the staged hash above. Applied validation receipt
SHA256`577eeff1f8761a3b5e8c048ef96ab867729a685ed781a312fc5f3c1591dd2146`
retains278 passing tests and the completed Qbert prerequisite hashes.

A prepared matched-budget Pong trial keeps B16/T16,4M/RGB48/Auto/N8/H15/R32,
coefficient.01, full18, no aid,500k actions/62,454 updates and the45-minute cap,
changing only the three native rates. Own actual initial must match all346
control tensors. Fixed final/initial24-natural-match evaluation retains the
mean15/90%-win gate,600k-action/300s caps, zero updates/truncations and full
replay/tensor/guard audits. Four recovery checks remain at100k intervals.

Controller compilation, complete queue normalization against the completed B16
control and unselected refusal pass. The CPU comparison reader validates the
completed control; candidate comparison waits for completed audits and curves.
Artifacts:`/x/Temp/kindle-cheap-20261025/pong-4m-48-b16-r32-intrinsic001-lr2-500k`;
controller:`run_pong_b16_lr2_500k.py`, SHA256
`9e5d36bad5bd0b82f695caf8d68546795504612b4d30459f432a91546b595cf8`;
prepared declaration`0fcaab73ca3dfb2e305dff98ab09ffd6b924091ac1e5229c4e5d5b0d76e53dfc`;
queue`3acdd0851b14f3fa1bd8f62ada99af8ede2d95a71d2379edf69576824d1fc731`.

The selected trial pins the applied CLI validation and committed runner source,
completed Qbert review and the original B16 control. Its controller requires
those pins and a terminal preceding service. Same500k/62,454-update budget,
15,988,224 sampled positions and first376; no new shape check or native build.
Audit1200s/service4560s; any cutoff is retained without cap increase or resume.
Rate rescaling is a hypothesis on one reused seed, not independent confirmation
or equivalence to B8. No extension or automatic next allocation is selected.

Started **11:18:38 UTC**, source`d4d1093`, native`bfb8c31`, CLI`22bd61b`.
Startup verifies the complete config against the B16 control with only the three
rate changes,3,578,225 parameters, zero counters and first376:1/384:2/392:3.
All346 finite initial tensors match the control. Initial metadata SHA256
`8e9b00dce6b9292e4813ec652c2fac47d5123d85802006ea5f73a933114dc15c`;
selected declaration`0d2bded744d86db073ea3ea34be6bb58d1ba600eb2fd8bd26bc4fedc03160ec6`.
Four recovery audits, frozen evaluations and completed curves/comparison are
automated. Frozen learning results remain pending.

The100k recovery audit passes at12,454 updates, with346 finite tensors and12,554
strict finite reports through100,808 actions. Captured98,304-action progress
is449.955s with zero debt. Recovery metadata SHA256
`367b440f527105c9f480d586c64b69d70e2b5e2e400b098f7dc4e50b48f8aec5`.
This is numerical/counter evidence; fixed-final frozen competence remains pending.
