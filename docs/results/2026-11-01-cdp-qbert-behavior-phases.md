# Qbert behavior-rate mean hides a first-pyramid failure and mixed episode changes

The doubled actor/critic rate leaves the official greedy mean nearly unchanged,
but individual outcomes differ substantially. Candidate lane1/episode2 scores
**425** without clearing the first pyramid,versus15,700 in that control slot.
The other23 slots gain15,725 points in total;the failed slot loses15,275.
The net450 points across24 episodes gives the observed **+18.75 mean**.
Twelve of the other23 slots improve and11 worsen. Identical slot indices do
not make these causal paired interventions:prior game and posterior RNG
histories can differ.

| Model / mode | Official mean,all24 | First pyramids | Mean later reward,all24 | Mean later actions,all24 |
|---|---:|---:|---:|---:|
| Base rate / greedy |14,339.583|24/24|13,789.583|1,899.542|
| Behavior8e-5 / greedy |14,358.333|23/24|13,813.542|1,766.208|
| Base rate / sampled |13,761.458|24/24|13,211.458|1,644.625|
| Behavior8e-5 / sampled |14,173.958|24/24|13,623.958|1,702.167|

All four means still miss15k. The candidate's23 successful greedy episodes
average14,964.130,also below15k,but this conditional subset is not an alternative
score or gate. Successful candidate episodes in both modes reach the milestone
at frame1,219 with550 points through its crossing action. The failed episode
has648 actions/2,592 frames and no milestone;its425 points stay in the unreached
prefix,withzero later reward/work. Every selected episode remains in official
means and the original first-three-per-lane selection.

Late replay imitation CE over actions150k–200k rises **.051059703→.055250033
(+8.21%)**,combining the two existing25k windows by their loss sums and label
weights. Both have4,687 reports and240 potential rows/update. This is replay
fit,not held-out action accuracy,and the policies visited different training
states. It supplies no evidence that doubling the rate improved imitation fit.

The failed greedy episode's first different command versus the control slot
is action51,ending at relative frame204:RIGHT versusNOOP. The first50 logged
commands,rewards and frame increments agree. This does not prove matching
observations:ordinary frozen logs contain no live pixel hashes. Against the
other23 successful candidate greedy episodes,first command differences occur
at42 (four episodes),51 (12),61 (two),and198 (five). Thus five successful
episodes share the failed episode's first197 commands. A single early command
difference is not an identified cause or a justified hand-coded correction.

The result supports retaining the better-supported base behavior rate and the
separately qualified faster teacher;it does not select another rate,loss,
controller,confirmation learner or automatic successor. The25k-window losses,
all96 episode rows,per-lane means,conditional denominators and command-prefix
comparisons remain available. The existing Breakout B8/T32 learner continues.

## Evidence and scope

This exploratory check was selected after inspecting the known425-point failure.
It computes48 new candidate phase rows and reuses48 already completed control
rows from the [H5/R24 seed diagnostic](2026-11-01-cdp-qbert-h5-r24-phases.md).
No old phase job is rerun. Candidate phases reconcile against saved logs and
verified replays;control rows preserve their original hashes and reconcile
against the completed assessments. Every official mean,success count,selected
ID and relevant action/frame/return total matches. The crossing action remains
intact;later begins with its successor and is not a general game-level label.

Eight boundary/absent-milestone andsix command-divergence fixtures pass.
All22 diagnostic inputs and292 active Breakout inputs remain unchanged,with
originalPID812861/invocation258e9d9deb0b463a91d4206a198085e4 before/after.
Selection takes1.539s wrapper;analysis **1.719s/1.763s wrapper**,completing
**16:06:19 UTC November1**,sourcec197078. One low-priority CPU core,stdlib only,
zero native/emulator imports,game replay,GPU work,updates,failures or retries.
Retain CPU co-host time without correcting Breakout's training time.

[Fixed protocol](../experiments/2026-11-01-cdp-qbert-behavior-phases.md),
[completed behavior-rate comparison](2026-11-01-cdp-qbert-behavior8e5.md).
Evidence`/x/Temp/kindle-qbert-behavior-phases-20261101`;complete,do not rerun.
The five-game goal remains unfinished.

| Artifact | SHA256 |
|---|---|
| Declaration |`9522fbab60b80162ad87d5ff1390de1fc6c2d8ecf2c848e8d9733b0b6f756bc6`|
| Review |`2d9c06d7a19402e11f1a4487436f9ff35104949fe29cc80be949b3058470fccd`|
| All96 episode rows |`106be57a2f17ae60aa512181c42fad04d0b558edad1338a10206660c8df6a033`|
| Failure detail |`8b5a3770a4a3f0401bbf160c6cc0ebd94697f7ffc52c925c2b6693e2d1901303`|

