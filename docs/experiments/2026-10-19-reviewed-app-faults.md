# Exact historical application-fault review

The upstream-refresh guard stopped at 05:11 UTC **before spawning a child**.
The same boot's journal contains nine Xid records on October 18, 16:21–21:46
UTC: six Xid 32, two Xid 31 and one Xid 69, naming `Ronin.exe` and
`PartyHardGame.e`. All five named PIDs have exited. No subsequent Xid, GPU reset
request, GSP timeout or PCIe AER appears in the reviewed journal. This does not
prove device health or identify the faults' cause.

[NVIDIA's current catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
assigns application restart to these codes, unlike reset/reboot-required codes.
The user's standing direction is to proceed unless wedged; no driver reset,
reload, reboot or other recovery is authorized or performed here.

Keep the first stopped guard and all original helpers/declarations untouched.
A separately copied, tested host guard may accept **only the exact nine
cursor/message pairs at baseline**, on the same declared boot/driver, checking
that each named PID remains absent. Every new occurrence (including identical
text with a different cursor), any live fault, other Xid or device failure still
stops. The full journal and reviewed events remain in the evidence; there is no
global Xid ignore flag. The known-good `e45b9c43` control is first, under the
unchanged bounded refresh workload. Its success would establish that workload's
execution, not hardware faultlessness. Failure stops for review.

Only the never-started RGB/CDP processes are re-declared. No benchmark sample,
native binary, numerical condition, timeout or memory limit is changed. No
NVML polling. The isolated guard/tests and exact records are under
`runs/upstream-refresh-20261019.KWEh8dEH/{guard_runtime,host-review.json}`.
