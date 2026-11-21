# Terminal-weight discrimination and fixed inverse diagnostic

The completed weight10 learner has weaker frozen gameplay than the retained
control. Its matched arrivals show much lower terminal prediction error and higher
error at intermediate life losses. Select one CPU-only analysis of those saved
outputs to separate ranking from probability levels. No model/emulator execution,
checkpoint change, fitted calibration, threshold search or new learner occurs.

Use all120 matched arrivals:24 game endings and96 intermediate life losses,
three/twelve per each of eight lanes. Join endpoint/outcome identities exactly.
Primary is posterior ROC AUC; one-step prior is secondary. Lower continuation
ranks a game ending first. Count all2,304 terminal/nonterminal pairs, half credit
for exact ties. Report paired candidate-minus-control AUC and its95% interval
from5,000 whole-lane bootstrap draws, seed20261104, recomputing pooled AUC using
lane multiplicities. Report order reversals and the already fixed below-.5 counts.

Separately apply exactly one analytic transformation to weight10 outputs. Let
gamma be the actual F32 continuing target, F32(1)-F32(1)/F32(333), and w=10. If
p is conditional continuing probability, minimizing the weighted soft-target BCE
gives q=gamma*p/(p+w*(1-p)). Its inverse to the unweighted soft target is

    f(q) = gamma*w*q / (gamma+(w-1)*q).

This is a conditional optimum identity, not a calibration guarantee for a finite
learned network. Apply the expression unchanged to every saved q in[0,1]; report
how many q exceed gamma, where this becomes algebraic extrapolation. Do not clip,
fit or sweep. The expression is strictly increasing, so verify all pair orders
are preserved, including ties. It cannot improve ROC AUC.

For terminal and intermediate-life groups separately, report means, MSE to their
actual targets, below-.5 counts and paired MSE differences with the same lane
bootstrap: raw candidate minus control, inverse candidate minus control and
inverse candidate minus raw candidate. Do not use the cohort's20% terminal
prevalence as a population estimate or combine groups into a headline error.
No replacement loss or behavior correction is selected. Altering a diagnostic
continuation output does not alter the already learned actor.

Reuse the completed arrival declaration's1,085 pins and44 final artifacts, plus
its successful independent closure. Pin this protocol, analysis and tests before
outcome analysis. Test exact AUC cases, lane resampling, endpoint validation,
monotonicity, the conditional-optimum round trip, gamma endpoints, w=1 identity,
invalid inputs and forbidden native/emulator imports. Run once on one CPU core
at nice10 under60s. Independently recount pairs, transformed outputs, group errors
and intervals from saved inputs and verify final artifacts. Preserve any failures;
do not repeat completed stages or change tolerances after seeing outcomes.

Root `/x/Temp/kindle-terminal-weight-ranking-20261104`. The source captures use
different native versions and learned states; posterior outputs are offline F64
estimates, with native-checked prior outputs. This reused control-policy cohort
cannot identify a causal gameplay bottleneck or candidate-policy calibration.
Retain143.667 greedy/70.750 sampled, all0/24 two-wall. The original864/90%-two-wall
and five-game goal remain unfinished.
