struct Params {
    starts: u32, horizon: u32, batch: u32, length: u32,
    bins: u32, actions: u32, intrinsic: u32, padding: u32,
    lambda: f32, discount: f32, bonus_scale: f32, rate: f32,
    rows: u32, offset: u32, padding2: u32, padding3: u32,
}
var<uniform> params: Params;
var<storage, read> values: array<f32>;
var<storage, read> bins: array<f32>;
var<storage, read_write> encoded: array<f32>;
var<storage, read_write> status: atomic<u32>;
var<workgroup> indices: vec2<u32>;
var<workgroup> fractions: vec2<f32>;

@compute @workgroup_size(64)
fn encode(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.x;
    if lane == 0u {
        if !(abs(values[params.offset + row]) <= 3.402823466e38f) { atomicOr(&status, 16u); }
        let value = clamp(values[params.offset + row], bins[0], bins[params.bins - 1u]);
        var low = 0u;
        var high = params.bins;
        while low < high {
            let middle = low + (high - low) / 2u;
            if bins[middle] < value { low = middle + 1u; }
            else { high = middle; }
        }
        if low == 0u || low == params.bins {
            indices = vec2<u32>(min(low, params.bins - 1u), params.bins);
            fractions = vec2<f32>(1.0f, 0.0f);
        } else if bins[low] == value {
            indices = vec2<u32>(low, params.bins);
            fractions = vec2<f32>(1.0f, 0.0f);
        } else {
            let distance_low = abs(value - bins[low - 1u]);
            let distance_high = abs(bins[low] - value);
            let total = distance_low + distance_high;
            indices = vec2<u32>(low - 1u, low);
            fractions = vec2<f32>(distance_high / total, distance_low / total);
        }
    }
    workgroupBarrier();
    for (var column = lane; column < params.bins; column += 64u) {
        encoded[row * params.bins + column] = select(0.0f, fractions.x, column == indices.x)
            + select(0.0f, fractions.y, column == indices.y);
    }
}
