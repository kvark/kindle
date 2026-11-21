# Repair group identity in the CPU prefix report

The first prefix analysis fails at18:03:55 UTC November1 after5.920s wrapper
time. It creates no review. Its dictionary uses tensor names without checkpoint
groups:357 arrays have324 distinct bare names,including22 names shared across
groups. The final357-entry assertion correctly rejects the incomplete report.
The source,15 original pins,declaration,start receipt and failed execution stay
unchanged under `/x/Temp/kindle-qbert-budget-prefix-20261101`.

Select a separate CPU repair under `group-key-repair`. Change only
`tensors[key]` to `tensors[group+'/'+key]` in the report accumulator. Keep the
same logs,200k endpoint,event filters,all357 array comparisons,120s cap,2GiB
limit and active-process checks. No GPU work or learner mutation is allowed.
The failed analysis must reread its CPU inputs because no review was saved;
no successful GPU,qualification or completed comparison stage is repeated.

Qualify the exact one-line source difference,the actual accumulator expression
on repeated-name synthetic groups,and all357 declared schema entries. Reuse the
14 unchanged numerical/event fixtures without rerunning them. Pin both original
and repaired sources,original failure receipts,qualification and this protocol.
Repaired completion must explicitly preserve the failed original attempt.

The learner remains PID826399/invocation6947a8ed4d734296a1cff8e1af8113b6.
Its200k/18,715 recovery succeeds18:03:18 UTC with357 finite tensors and all four
files retained. Do not restart it or alter its final assessments. No change to
the comparison question or acceptance of divergent prefixes is made.
