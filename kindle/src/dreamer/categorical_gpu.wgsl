struct Params {
    bins: u32,
    rows: u32,
    output_offset: u32,
    _pad: u32,
}

var<storage, read> logits: array<f32>;
var<storage, read> bins: array<f32>;
var<storage, read_write> means: array<f32>;
var<uniform> params: Params;
var<workgroup> maxima: array<f32, 128>;
var<workgroup> sums: array<vec2<f32>, 128>;

// Stable 1-exp(-x) for close mirrored logits; subtracting rounded exp values
// would destroy small reward/value predictions on the exponential support.
fn exponential_difference(x: f32) -> f32 {
    if x < 0.125 {
        return x * (1.0 + x * (-0.5 + x * (1.0/6.0 + x * (-1.0/24.0
            + x * (1.0/120.0 + x * (-1.0/720.0 + x/5040.0))))));
    }
    return 1.0 - exp(-x);
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    let row = group.x;
    if row >= params.rows { return; }
    let base = row * params.bins;
    var maximum = -3.402823e38;
    for (var i = lane; i < params.bins; i += 128u) {
        maximum = max(maximum, logits[base + i]);
    }
    maxima[lane] = maximum;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride /= 2u) {
        if lane < stride { maxima[lane] = max(maxima[lane], maxima[lane + stride]); }
        workgroupBarrier();
    }
    maximum = maxima[0];
    let middle = params.bins / 2u;
    var total = vec2<f32>(0.0);
    for (var lower = lane; lower < middle; lower += 128u) {
        let upper = params.bins - 1u - lower;
        // Match the CPU's F32 shift before its F64 exponential.
        let a = logits[base + lower] - maximum;
        let b = logits[base + upper] - maximum;
        let high = exp(max(a, b));
        let delta = abs(a - b);
        let difference = high * exponential_difference(delta);
        let signed_difference = select(-difference, difference, b >= a);
        total += vec2<f32>(signed_difference * bins[upper], exp(a) + exp(b));
    }
    if lane == 0u { total.y += exp(logits[base + middle] - maximum); }
    sums[lane] = total;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
    if lane == 0u { means[params.output_offset + row] = sums[0].x / sums[0].y; }
}
