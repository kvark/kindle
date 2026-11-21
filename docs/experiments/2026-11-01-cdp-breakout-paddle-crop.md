# Breakout paddle crop: one fixed CPU pixel measurement

The completed pooling diagnostic excludes6,545 of42,303 selected teacher
decisions because the raw paddle component touches the horizontal crop. The
teacher measures red pixels in rows185:198,columns8:152;its response model clips
positions to15.5–147.5. Whether the crop biases observed centers at real edges
has not been measured on full-width pixels.

Select one new annotation replay of the retained unchanged teacher trajectory:
52,072 recorded actions/208,254 raw frames,with the original first3 natural
episodes per8 lanes (24 episodes) as primary descriptive population. Reproduce
the original actions,debug,pixel hashes,rewards,frames,boundaries,episodes and
raw RGB aggregate hash. Compare every prior cropped raw annotation exactly.
No new policy trajectory,learning,native import,GPU allocation or gameplay
score is selected;the source remains225.083 andzero two-wall successes.

Add a full-width detector using the same red mask and rows185:198,columns0:160.
Keep the original largest unique contiguous run rule. Record absent,tied,
nonstandard and image-edge components;do not hide exclusions. For raw paired
comparisons accept uncropped full-width components of width12 or16. For pooled
observations require both last raw components accepted,same width/row,and the
full pooled centroid to equal their midpoint within1e-9. Initial/reset images
lack that pair and remain separate. Compare the actual logged cropped teacher
center against this full pooled centroid,not against a newer raw time target.

Report selected/all row counts,center differences,component widths and extreme
positions,per-lane counts and fixed near-descending/impact-within-one-action
subsets. Count stationary full centers satisfying the existing x≤16/x≥147
threshold when the original cropped observation does not;require valid adjacent
full observations for velocity and retain denominators. These are observation
comparisons,not proof of a causal missed collision or a selected edge reset.

CPU fixtures cover empty/ambiguous masks,interior and clipped12/16-pixel runs,
both edges,and valid/invalid pooled pair alignment. Pin the prior replay and
detector sources,ROM/emulator/wrapper,local diagnostic and current Qbert
declaration/launch. Keep the active Qbert encoder8e5 learner's528 inputs and
PID831942/invocationbae3fe2dd4a24070ab2845a7828145fa unchanged before/after.
Use one low-priority CPU core,300s replay/360s controller caps,retain co-host
time without correcting Qbert training time. Never restart the active learner.

Preserve failures and originals. No detector integration,response refit,
teacher gameplay comparison,loss,new learner,cap increase or automatic successor
is selected. Measurement precision,ball/paddle overlap,pooled timing and reused
teacher trajectories limit interpretation. Root
`/x/Temp/kindle-breakout-paddle-crop-20261101`;five-game goal unfinished.

CPU preflight passes18:40:12 UTC November1 in3.309s:eight mask/pair fixtures,
prior source/pixel-annotation pins and ROM/emulator/wrapper identity pass.
No native module or environment is constructed. All528 active Qbert pins and
originalPID831942/invocationbae3fe2dd4a24070ab2845a7828145fa remain unchanged.
Retain this CPU co-host time without correcting Qbert wall time.
