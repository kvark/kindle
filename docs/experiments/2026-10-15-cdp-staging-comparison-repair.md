# Repair the CDP staged-world observer comparison

Declared October15,2026, after reviewing the stopped world process in
`runs/cdp-bf16-qualification-20261015.SGFYYAOP`. No numerical gate is relaxed.

The independent BF16 loss/projection boundary passes all5,585,254 values.
The actual-world test then fails comparing an in-memory `serde_json::Value`
with a deserialized JSON value before staged backward begins. Independent CPU
streaming of both retained assertion operands verifies all5,618,720 F32 values
bit-exact across nine fields, while their F64 parser representations differ.
Its first CPU log extractor failed on assertion-label whitespace; the corrected
extractor and both sources remain. This is an observer serialization mismatch,
not an observed model-output mismatch. Original GPU failure/logs remain failed.

Use a separate source/binary copy. Persist first outputs and metrics before
comparison, then deserialize both observed/reference files through the same
parser. Keep exact equality and saved-tensor checks; add a CPU round-trip
regression. No model, backend, compiler, precision, optimizer or input change.
This also retains compact assertion messages rather than printing millions of
numbers on failure; full observed files remain available.

Run only the previously uncompleted actual-world qualification: one initializer,
original whole/staged and changed-late whole/staged backwards, then exact
original restoration. Same declarations for poisoned late inputs, every raw
gradient, unchanged346 saved tensors, graph/allocations/dependencies and state.
The passed independent boundary is retained without a GPU repeat. No timing,
learning or production promotion. This is not a full-model independent oracle.

One separately guarded GPU process,180s, persistent service15min bound,
Restart=no,KillMode=control-group; expected5080/580.178.04/boot,>=2GiB sampled
Vulkan estimated headroom. Record standalone allocation warnings. Stop on
API/numerical/hard/deadline failures for review. No NVML polling/recovery.
CPU build/preparation/review1CPU/2GiB/no swap; no GPU-worker CPU quota.
