struct Params {
    starts: u32, horizon: u32, batch: u32, length: u32,
    bins: u32, actions: u32, intrinsic: u32, padding: u32,
    lambda: f32, discount: f32, bonus_scale: f32, rate: f32,
    rows: u32, offset: u32, padding2: u32, padding3: u32,
}
var<uniform> params: Params;
var<storage, read> returns: array<f32>;
var<storage, read_write> state: array<f32>;
const SORT_LENGTH: u32 = _SORT_LENGTH_u;
var<workgroup> sorted: array<u32, SORT_LENGTH>;

fn order_key(bits: u32) -> u32 {
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn percentile(q: f32) -> f32 {
    let position = q * f32(params.starts * params.horizon - 1u);
    let low = u32(floor(position));
    let high = u32(ceil(position));
    let mix = position - f32(low);
    return bitcast<f32>(sorted[low]) * (1.0f - mix) + bitcast<f32>(sorted[high]) * mix;
}

@compute @workgroup_size(256)
fn normalize(@builtin(local_invocation_index) lane: u32) {
    for (var i = lane; i < SORT_LENGTH; i += 256u) {
        var bits = 0x7f800000u;
        if i < params.starts * params.horizon { bits = bitcast<u32>(returns[i]); }
        sorted[i] = bits;
    }
    workgroupBarrier();
    for (var width = 2u; width <= SORT_LENGTH; width *= 2u) {
        for (var distance = width / 2u; distance > 0u; distance /= 2u) {
            for (var i = lane; i < SORT_LENGTH; i += 256u) {
                let other = i ^ distance;
                if other > i {
                    let a = sorted[i];
                    let b = sorted[other];
                    if (order_key(a) > order_key(b)) == ((i & width) == 0u) {
                        sorted[i] = b;
                        sorted[other] = a;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    if lane == 0u {
        state[0] = (1.0f - params.rate) * state[0] + params.rate * percentile(0.05f);
        state[1] = (1.0f - params.rate) * state[1] + params.rate * percentile(0.95f);
    }
}
