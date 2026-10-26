# Pong learning-rate candidate after the B16 tradeoff

**CLI preparation only; no GPU allocation selected.** The Qbert12M/RGB48/B16
200k screen remains active. Do not modify its pinned production runner until
training, both frozen arms, audits and completed-run analysis have finished.

PongB16 uses22.37% less training time at500k actions but scores11.875 versus B8's
14.958, both24/24 wins and below mean15. B16 takes62,454 optimizer steps versus
124,939, with nearly equal sampled replay positions. This motivates a learning-
rate candidate; it does not prove that fewer steps caused the lower score.

The existing native configuration supports base, encoder and dynamics rates.
`configure_d3_optimizer` applies encoder/base and dynamics/base multipliers;
behavior defaults to the base rate. The CLI currently fixes CDP at4e-5/6e-6/4e-4.
A staged change permits the existing `--learning-rate` on fresh CDP runs and
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
The production runner remains`8b8491e75acac43eedfa48ce53b1bd179e91c9c2277d02169628bd4e7c470e9b`.

After the current Qbert result is reviewed, decide whether to apply this exact
patch, checking both before hashes, then validate the applied files and repin
future declarations. A possible matched-budget Pong test would keep B16/T16,
4M/RGB48/Auto/N8/H15/R32, coefficient.01, full18, no aid,500k actions and the
45-minute cap, changing only the three native rates. It would require own actual
initial, matched initial tensors, fixed final/initial24-natural-match evaluation,
the unchanged mean15/90%-win gate and full audits. That learning trial has no
controller/declaration or queue yet; no automatic allocation follows from these
CPU tests, and no learning outcome is claimed.
