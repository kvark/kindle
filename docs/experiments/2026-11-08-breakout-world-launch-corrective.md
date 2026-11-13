# Breakout: bounded corrective replay with clean launches

**Closed; retention fails.** [Measured results](../results/2026-11-08-breakout-world-launch-corrective.md).

Select one4,096-update corrective phase after the [clean-wait collection](../results/2026-11-08-breakout-world-clean-launch.md)
and conditional support review. Start from the closed small64 final16,384 and
finish at20,480. New data and sampler change deliberately; this is a warm start,
not uninterrupted training,an independent learner,an isolated data effect or a
capacity comparison. No completed learner/collector/assessment is restarted.
Root `/mnt/data/kindle-breakout-world-launch-corrective-20261108`.

Use unchanged v18 native/source/runner7f797a2b,SHA38680c73,size1M/RGB64,
603,915 weights/80 logical parameters/240 tensors,B8/T32/H16/full BPTT32,Auto,
rates4e-4,RGB100,KL1/.1/free-nat1,event scales1,mixing.01. Retain all optimizer
moments/ages;the checkpoint omits live belief/RNG and restore resets those.
Native Meganeura/Blade performs all learning/inference. No actor/critic,CDP,
object/RAM supervision,new primitive,build or new backend qualification.

Freeze the full4,096-row plan before allocation. Per minibatch use two original
natural windows,one supplement natural,one existing wait window,one existing
terminal window,two new clean-launch windows andone retained paired branch.
Launch slot1 cycles all288 branches at start16:the first imagined command
follows16 observed clean-wait arrivals. Slot2 cycles a fixed permutation at
start32:observe16 launched arrivals and target decisions17–32. Retained paired
slot rotates contact/life_loss/ending and samples all corresponding branches
without future-outcome selection. Use their existing prefixes and short terminal
tails,never new collection or padded truth. Labels select fitting data only.
All eighteen commands and both repetitions remain;related/duplicate histories
are disclosed. Onlyfour distinct16-arrival launch histories exist.

CPU preparation verifies native/config/source identity,all data manifest hashes,
held-out-source seed separation,arrival-aligned windows,all slot constraints,
every branch's inclusion and expected finite/reset-valid/terminal/reward
exposures for this new plan. Do not recount an old generic plan ledger or repeat
old native qualification. Existing unchanged-shape qualification is reused.
The first actual phase update counts toward4,096;existing finite13-report,
mask and weighted-total checks apply and stop on failure. Check live services
and use the existing guarded queue;stop on real faults,never reset the driver.

Save inherited initial state and require exact240-tensor equality to small64
final. Reuse that model's already sealed frozen actual-initial evaluations;
do not query it again. Keep the original natural fitting/development selection,
fixed checkpoints atnew2,048/final4,096 and curve evaluations every1,024.
Final dense256 fitting/128 development and exact18-array restore checks apply.
No best-checkpoint selection,automatic extension or additional learner.

After successful learning/CPU review,separately pin the final-only84-group
chosen-action assessment. Same42 plans,histories16 primary/32 secondary,eight
draws,endpoints1/4/8/16/32. Reuse cached controls and canonical oracle outcomes;
do not repeat old forecasts. Keep every action/horizon/category/individual alarm,
regression,ball/paddle/reward/end/brick metric. Compare both H0 and the inherited
small64;test original unchanged gates,not just launch accuracy or loss decrease.
Require exact source/frozen tensors,finite losses,counts,restore and terminal
service/host-guard checks before interpretation. Save plots and fixed first-draw
clips if useful. All logs/weights/data remain outside Git;publish a compact
result and update PR31. Intermediate gates cannot finish the full goal without
precise physics,event timing/calibration,persistent removed-brick memory,late/
tunnel/second-wall/above-top coverage,untouched families and fresh confirmation.
