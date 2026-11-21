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
var<storage, read> replay: array<f32>;
var<storage, read> returns: array<f32>;
var<storage, read> weights: array<f32>;
var<storage, read> state: array<f32>;
var<storage, read_write> report: array<f32>;
var<storage, read_write> status: atomic<u32>;
var<workgroup> partials: array<f32, 5120>;

@compute @workgroup_size(256)
fn summarize(@builtin(local_invocation_index) lane: u32) {
    var sums: array<f32, 20>;
    let rows = params.starts * params.horizon;
    let scale = max(1.0f, state[1] - state[0]);
    var okay = true;
    for (var row = lane; row < rows; row += 256u) {
        var extra = 0.0f;
        if params.intrinsic != 0u { extra = params.bonus_scale * bonus[row]; }
        sums[0] += rewards[row + params.starts] + extra;
        sums[1] += extra;
        sums[3] += returns[row];
        let advantage = (returns[row] - values[row]) / scale;
        sums[4] += abs(advantage);
        sums[5] += abs(weights[row] * advantage);
    }
    for (var row = lane; row < rows + params.starts; row += 256u) {
        sums[2] += continuation[row];
    }
    for (var row = lane; row < params.starts; row += 256u) {
        var extra = 0.0f;
        if params.intrinsic != 0u { extra = params.bonus_scale * replay_bonus[row]; }
        okay = okay && abs(extra) <= 3.402823466e38f && extra >= 0.0f;
        sums[6] += extra;
        let prediction = rewards[row];
        let observed = replay[row];
        sums[7] += prediction;
        sums[8] += observed;
        sums[9] += abs(prediction - observed);
        if observed > 0.0f { sums[10] += prediction; sums[14] += 1.0f; }
        else if observed < 0.0f { sums[12] += prediction; sums[16] += 1.0f; }
        else { sums[11] += prediction; sums[15] += 1.0f; }
        if observed != 0.0f { sums[13] += prediction; }
        let live = (1.0f - replay[2u * params.starts + row]) * params.discount;
        sums[17] += continuation[row];
        sums[18] += live;
        sums[19] += abs(continuation[row] - live);
    }
    for (var channel = 0u; channel < 20u; channel++) {
        okay = okay && abs(sums[channel]) <= 3.402823466e38f;
        partials[channel * 256u + lane] = sums[channel];
    }
    if !okay { atomicOr(&status, 8u); }
    workgroupBarrier();
    for (var distance = 128u; distance > 0u; distance /= 2u) {
        if lane < distance {
            for (var channel = 0u; channel < 20u; channel++) {
                let index = channel * 256u + lane;
                partials[index] += partials[index + distance];
            }
        }
        workgroupBarrier();
    }
    if lane == 0u {
        let count = f32(rows);
        let replay_count = f32(params.starts);
        let positive = partials[14u * 256u];
        let zero = partials[15u * 256u];
        let negative = partials[16u * 256u];
        report[0] = partials[0u] / count;
        report[1] = partials[1u * 256u] / count;
        report[2] = partials[6u * 256u] / replay_count;
        report[3] = partials[2u * 256u] / f32(rows + params.starts);
        report[4] = partials[3u * 256u] / count;
        report[5] = scale;
        report[6] = partials[4u * 256u] / count;
        report[7] = partials[5u * 256u] / count;
        report[8] = partials[7u * 256u] / replay_count;
        report[9] = partials[8u * 256u] / replay_count;
        report[10] = partials[9u * 256u] / replay_count;
        report[11] = partials[13u * 256u] / max(1.0f, positive + negative);
        report[12] = partials[11u * 256u] / max(1.0f, zero);
        report[13] = positive + negative;
        report[14] = partials[10u * 256u] / max(1.0f, positive);
        report[15] = report[12];
        report[16] = partials[12u * 256u] / max(1.0f, negative);
        report[17] = positive;
        report[18] = zero;
        report[19] = negative;
        report[20] = partials[17u * 256u] / replay_count;
        report[21] = partials[18u * 256u] / replay_count;
        report[22] = partials[19u * 256u] / replay_count;
        report[23] = state[0];
        report[24] = state[1];
    }
}
