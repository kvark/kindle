struct Params {
    starts: u32, horizon: u32, batch: u32, length: u32,
    bins: u32, actions: u32, intrinsic: u32, padding: u32,
    lambda: f32, discount: f32, bonus_scale: f32, rate: f32,
    rows: u32, offset: u32, padding2: u32, padding3: u32,
}
var<uniform> params: Params;
var<storage, read> rewards: array<f32>;
var<storage, read> values: array<f32>;
var<storage, read> continuation: array<f32>;
var<storage, read> bonus: array<f32>;
var<storage, read> replay_bonus: array<f32>;
// Three time-major arrays: observed reward, is_last, is_terminal.
var<storage, read> replay: array<f32>;
var<storage, read_write> returns: array<f32>;
var<storage, read_write> weights: array<f32>;
var<storage, read_write> replay_returns: array<f32>;
var<storage, read_write> replay_weights: array<f32>;
var<storage, read_write> continuation_target: array<f32>;
var<storage, read_write> world_weight: array<f32>;
var<storage, read_write> status: atomic<u32>;

fn valid(x: f32) -> bool { return abs(x) <= 3.402823466e38f; }

// Keep cancellation residuals local; every externally visible return stays F32.
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let total = a + b;
    let from_b = total - a;
    let error = (a - (total - from_b)) + (b - from_b);
    return vec2<f32>(total, error);
}

fn pair_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let sum = two_sum(a.x, b.x);
    return two_sum(sum.x, (sum.y + a.y) + b.y);
}

fn pair_scale(a: vec2<f32>, b: f32) -> vec2<f32> {
    let high = a.x * b;
    let low = fma(a.x, b, -high) + a.y * b;
    return two_sum(high, low);
}

@compute @workgroup_size(64)
fn imagined(@builtin(global_invocation_id) id: vec3<u32>) {
    let start = id.x;
    if start >= params.starts { return; }
    var next = vec2<f32>(values[params.horizon * params.starts + start], 0.0f);
    var okay = valid(next.x);
    for (var t = params.horizon; t > 0u; t--) {
        let arrival = t * params.starts + start;
        let row = arrival - params.starts;
        var reward = vec2<f32>(rewards[arrival], 0.0f);
        if params.intrinsic != 0u {
            let extra = pair_scale(vec2<f32>(bonus[row], 0.0f), params.bonus_scale);
            okay = okay && valid(extra.x) && extra.x >= 0.0f;
            reward = pair_add(reward, extra);
        }
        // Preserve the CPU's terminal conversion, including its F32 rounding.
        let terminal = 1.0f - clamp(continuation[arrival], 0.0f, 1.0f);
        let live = 1.0f - terminal;
        let bootstrap = pair_scale(pair_scale(vec2<f32>(values[arrival], 0.0f), 1.0f - params.lambda), live);
        let carried = pair_scale(pair_scale(next, params.lambda), live);
        next = pair_add(pair_add(reward, bootstrap), carried);
        returns[row] = next.x;
        okay = okay && valid(next.x) && valid(next.y) && valid(continuation[arrival]);
    }
    var cumulative = 1.0f;
    for (var t = 0u; t < params.horizon; t++) {
        let row = t * params.starts + start;
        cumulative *= clamp(continuation[row], 0.0f, 1.0f);
        weights[row] = cumulative;
        okay = okay && valid(values[row]) && valid(continuation[row]);
    }
    if !okay { atomicOr(&status, 1u); }
}

@compute @workgroup_size(64)
fn replayed(@builtin(global_invocation_id) id: vec3<u32>) {
    let batch = id.x;
    if batch >= params.batch { return; }
    let final_row = (params.length - 1u) * params.batch + batch;
    var next = vec2<f32>(returns[final_row], 0.0f);
    var okay = valid(next.x);
    for (var t = params.length - 1u; t > 0u; t--) {
        let arrival = t * params.batch + batch;
        let row = arrival - params.batch;
        var reward = vec2<f32>(replay[arrival], 0.0f);
        if params.intrinsic != 0u {
            let extra = pair_scale(vec2<f32>(replay_bonus[arrival], 0.0f), params.bonus_scale);
            okay = okay && valid(extra.x) && extra.x >= 0.0f;
            reward = pair_add(reward, extra);
        }
        let live = (1.0f - replay[2u * params.starts + arrival]) * params.discount;
        let trace = (1.0f - replay[params.starts + arrival]) * params.lambda;
        let bootstrap = pair_scale(pair_scale(vec2<f32>(returns[arrival], 0.0f), 1.0f - trace), live);
        let carried = pair_scale(pair_scale(next, trace), live);
        next = pair_add(pair_add(reward, bootstrap), carried);
        replay_returns[row] = next.x;
        replay_weights[row] = 1.0f - replay[params.starts + row];
        world_weight[row] = replay_weights[row] * (f32(params.length) / f32(params.length - 1u));
        okay = okay && valid(next.x) && valid(next.y);
    }
    world_weight[final_row] = 0.0f;
    for (var t = 0u; t < params.length; t++) {
        let row = t * params.batch + batch;
        continuation_target[row] = (1.0f - replay[2u * params.starts + row]) * params.discount;
        okay = okay && valid(replay[row]);
    }
    if !okay { atomicOr(&status, 2u); }
}
