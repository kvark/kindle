# Frozen noisy-residual input attenuation

The [retained-stage trace](../results/2026-11-09-breakout-reward32-stage-trace.md)
finds weak new-ball formation already athigh noise,withlater denoising improving
recovery. Test one frozen function perturbation toidentify sensitivity tothe
noisy-current residual channels,without another learning/build campaign.
Root `/mnt/data/kindle-breakout-noisy-channels0p1-20261109`.

Clone qualified reward32 checkpoint2048. Scale only first-convolution OIHW
coefficients forinput channels84/85/86 byfloat32.1:16 outputfilters ×three input
channels ×5×5 =1,200 coefficients. Input layout is12 past RGB+72 action planes+
three noisy residual planes+one sigma plane. History/action/sigma coefficients,
biases/norms,allother parameters,optimizer arrays andmetadata remain exact.
Original checkpoint staysimmutable. Onlyone of135 tensors changes;134 arrays
andall57 event tensors remain exact. Preserve SafeTensors metadata,including
adam_step2048 andlogical layout/order,andverbatim JSON header. Do not resume
training fromthis manually perturbed optimizer state;this isfrozen analysis only.

CPU preparation verifies flat OIHW shape/layout fromretained source,data hashes,
alluntouched entries/tensors,metadata,1,200 scaled entries andindependent float64
linear-equivalence within1e-6. The stored float32 product isnot claimed bit-exact
toalternative FMA/input scaling. Factor.1 isdeclared once,no sweep. This attenuates
neural conditioning onboth noise andpartly generated residual information atall
three stages;it isnot achange onlytoinitial noise,skip reconstruction,cin formula,
sigma schedule ortraining. It maycreate inference/training mismatch andcannot
prove another trained input-gain recipe works orfails.

Reuse exact native60b9b67244d7/source8dbf9ab5 andmatching runner,no production API,
network shape,loss,precision orbackend primitive change. Same RGB64,B8,Auto,
three Euler steps,draws83001/83003/83009/83023 and224 reused fitting arrivals.
No new learner,SGD,game actions/frames,object labels,actor/critic/CDP orbuild.

One guarded assessment restores the clone,queries four16-step ordinary rollouts
and108 actual-history one-step cases,retaining RGB/reward/positive/continuation/
observed-denoising outputs.135 own frozen tensors must match the clone,not the
original deliberately changed model. Second guarded stage usesqualified native
trace executable786ab275,checking108 finalbatch arrays/864 images and20 origin
arrays exactly against candidate outputs,then135 frozen tensors exact again.
Allthree intermediate denoised estimates retained. Candidate expectations enter
onlyparity checks,outside causal model generation. Baseline metrics/stages are
read fromqualified cached control,not rerun. Pin derived trace inputs separately.

Keep original gates/detector:ordinary supported1/4/8/16 allwithin1px,all16>=90%
of110 perdraw,actual-history one-step>=90% of198. Missing detections fail;18
unsupported endpoints per task/draw unscored. Retain allper-case/horizon,ball/
paddle/brick/pixel/false-detection/event andstage contrast/motion regressions.
No best-stage/horizon selection. Four draws share one model;one causal positive/
no terminals cannot establish calibration. This perturbation isnot learning or
automatic adoption ofa new model.

Pin source/native/checkpoint/clone/data/scripts before GPU,keep them immutable.
Use new filenames/declarations forCPU analysis. Existing AMD host guard,300s cap
eachstage,one service,no NVML/reset/reload/timeout restart. Require terminal
success,two sealed guards/18 files before CPU analysis. No build cache created.
Fullres protocol failure staysfailed;allold jobs stayclosed. No learner,extension,
collector orautomatic successor. Full physics/events,natural calibration,changed-
brick memory,late/tunnel/second-wall/above-top,untouched families andfresh
confirmation remain required;the full world-model goal remains unproven.

Completed: [negative frozen result](../results/2026-11-09-breakout-noisy-channels0p1.md). Original gain retained;218 closure pins andtwo guards pass.
