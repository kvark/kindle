struct Params {
    frame: vec4<u32>, // first centered value, first packed word, words, unused
    _pad0: vec4<u32>,
    _pad1: vec4<u32>,
    _pad2: vec4<u32>,
}

var<storage, read> values: array<f32>;
var<storage, read_write> packed: array<u32>;
var<uniform> params: Params;

// Learned-RGB preprocessing emits exact `byte / 255 - 0.5` values; four of
// them are kept as one little-endian word.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.frame.z { return; }
    var word = 0u;
    for (var k = 0u; k < 4u; k += 1u) {
        let value = values[params.frame.x + 4u * id.x + k];
        let byte = u32(clamp(round((value + 0.5) * 255.0), 0.0, 255.0));
        word |= byte << (8u * k);
    }
    packed[params.frame.y + id.x] = word;
}
