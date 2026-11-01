# Identical logged paddle inputs have different observed outcomes

The teacher's four response inputs do not determine the next observed paddle
position exactly in the retained data. In the episode-separated held-out set,
4,597 of 13,102 rows belong to repeated exact-input groups with multiple next
positions. Of these, 3,828 belong to groups spanning at least four pixels.
Near descending balls, the respective counts are 514 and 398 of 2,955 rows.
These are descriptive counts of rows in groups, not independent probabilities
of a model error or missed contact.

The [fixed diagnostic](../experiments/2026-11-01-cdp-breakout-response-ambiguity.md)
groups exact logged paddle position, observed displacement, current command and
previous command, including the existing stationary-edge reset. It performs no
rounding, binning, regression or trajectory collection. The preceding
[rejected refit](2026-11-01-cdp-breakout-response-refit.md) remains unchanged.

| Subset | Rows | Repeated groups / rows | Groups / rows with multiple outcomes | Groups / rows spanning ≥4px | RMS within-group pair difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held-out all | 13,102 | 2,237 / 10,177 | 890 / 4,597 | 763 / 3,828 | 2.845px |
| Held-out near descending | 2,955 | 475 / 1,533 | 157 / 514 | 125 / 398 | 2.512px |
| Held-out current interiors | 11,033 | 2,010 / 8,315 | 803 / 3,537 | 710 / 3,176 | 3.444px |
| Held-out stationary edge | 849 | 10 / 848 | 7 / 617 | 4 / 381 | 2.614px |
| All selected episodes0–2 | 37,448 | 4,458 / 33,483 | 2,011 / 18,564 | 1,761 / 16,198 | 2.866px |

Across repeated groups, median range is zero in every subset except stationary
edges (1.75px). Held-out-all and near-descending p95 range is10px.
RMS weights all unordered observation pairs within each group; frequent states
therefore dominate. Stationary edges contribute74,352 of106,330 held-out pairs.
Current-position interiors also show variation, so the observation is not
restricted to wall clipping. Full group histograms retain these denominators.

Hidden action state, sticky actions, pooled frame timing, other emulator state
and pixel measurement can all contribute. These measurements do not isolate
their contributions or establish an irreducible noise floor: richer inputs
could distinguish states that this grouping treats as identical. This teacher
diagnostic does not measure information in Kindle's recurrent state. It also
does not establish gameplay benefit for uncertainty-aware planning or another
loss. Shared seeds, teacher and selection history limit generalization.

The2.433s analysis/2.518s controller completes **10:42:08 UTC November1**,
source`20b1208`. A2.029s preflight checks four arithmetic fixtures and source
receipts; an independent.083s histogram recount checks20,221 groups across five
overlapping subsets with explicit distinct-position pair arithmetic. All seven
declared inputs and442 active Qbert pins remain unchanged, preserving
PID781585/invocation`539c22282c4e42c7a22bc150961ef963`. Retain CPU co-host timing.
Zero emulator/native imports, GPU work, updates, failures or repeated stages.
No controller, loss, learner or automatic successor is selected.

Evidence: `/x/Temp/kindle-breakout-response-ambiguity-20261101`.

- Declaration: `06dbd25a033a2d91ce98663ebe570e79964c3dcaeebf49df5593d3c43126b333`.
- Review: `ebc1b9e0ea60b27f876946925ddd2fc5b75566421cd8183a0b24804e111ff627`.
- Group histograms: `613c016f2ca343a94266c14fa7f61279d8d179d3474f782a92ed02091bc54faa`.
- Completion: `32aa283e9eb26f1175bd9500f0a751f31258c96aadedeb99a20f3fe78472783a`.
- Independent recount: `ddf94bc5c45941bb70576fdf4b397697ece58dfcaa04445c52b752342ea1e448`.

Complete: do not rerun analysis, preflight or recount. The five-game goal remains
unfinished.
