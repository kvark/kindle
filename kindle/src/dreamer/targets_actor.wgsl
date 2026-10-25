struct Params {
    starts: u32, horizon: u32, batch: u32, length: u32,
    bins: u32, actions: u32, intrinsic: u32, padding: u32,
    lambda: f32, discount: f32, bonus_scale: f32, rate: f32,
    rows: u32, offset: u32, padding2: u32, padding3: u32,
}
var<uniform> params: Params;
var<storage, read> returns: array<f32>;
var<storage, read> values: array<f32>;
var<storage, read> weights: array<f32>;
var<storage, read> actions: array<f32>;
var<storage, read> state: array<f32>;
var<storage, read_write> encoded: array<f32>;
var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(64)
fn actor(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= params.starts * params.horizon { return; }
    let advantage = (returns[row] - values[row]) / max(1.0f, state[1] - state[0]);
    let weighted = weights[row] * advantage;
    var selected = 0u;
    var okay = abs(weighted) <= 3.402823466e38f;
    for (var column = 0u; column < params.actions; column++) {
        let index = row * params.actions + column;
        let action = actions[index];
        encoded[index] = select(0.0f, weighted, action == 1.0f);
        selected += u32(action == 1.0f);
        okay = okay && (action == 0.0f || action == 1.0f);
    }
    if !okay || selected != 1u { atomicOr(&status, 4u); }
}
