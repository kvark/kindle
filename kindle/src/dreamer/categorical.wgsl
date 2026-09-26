struct Params {
    rows: u32,
    classes: u32,
    unimix: f32,
    greedy: u32,
    masked: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
var<storage, read> logits: array<f32>;
var<storage, read> draws: array<f32>;
var<storage, read> allowed: array<u32>;
var<storage, read_write> onehot: array<f32>;
var<storage, read_write> selected: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= params.rows { return; }
    let base = row * params.classes;
    var maximum = -3.402823466e38f;
    var best = 0u;
    var valid = true;
    var count = 0u;
    for (var c = 0u; c < params.classes; c++) {
        let x = logits[base + c];
        valid = valid && abs(x) <= 3.402823466e38f;
        if params.masked == 0u || allowed[base + c] != 0u {
            if count == 0u || x > maximum { maximum = x; best = c; }
            count++;
        }
    }
    valid = valid && count > 0u;
    var normalizer = 0.0f;
    for (var c = 0u; c < params.classes; c++) {
        var p = 0.0f;
        if params.masked == 0u || allowed[base + c] != 0u {
            p = exp(logits[base + c] - maximum);
        }
        onehot[base + c] = p;
        normalizer += p;
    }
    var total = 0.0f;
    var best_probability = -1.0f;
    for (var c = 0u; c < params.classes; c++) {
        var p = 0.0f;
        if params.masked == 0u || allowed[base + c] != 0u {
            p = (1.0f - params.unimix) * onehot[base + c] / normalizer
                + params.unimix / f32(count);
        }
        onehot[base + c] = p;
        total += p;
        if p > best_probability { best_probability = p; best = c; }
    }
    if params.greedy == 0u {
        let draw = draws[row] * total;
        var cumulative = 0.0f;
        for (var c = 0u; c < params.classes; c++) {
            if onehot[base + c] > 0.0f { best = c; }
            cumulative += onehot[base + c];
            if draw < cumulative { best = c; break; }
        }
    }
    selected[row] = select(-1.0f, f32(best), valid);
    for (var c = 0u; c < params.classes; c++) {
        // Propagate invalid posteriors into the policy, where the action-only
        // readback rejects them. Never silently turn a NaN into a legal action.
        onehot[base + c] = select(bitcast<f32>(0x7fc00000u), select(0.0f, 1.0f, c == best), valid);
    }
}
