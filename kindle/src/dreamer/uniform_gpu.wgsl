struct Params {
    key0: vec4<u32>,
    key1: vec4<u32>,
    counter_stream: vec4<u32>,
    len: u32,
    skip: u32,
    raw_words: u32,
    _pad: u32,
}

var<storage, read_write> dst: array<u32>;
var<uniform> params: Params;

fn rotate(x: vec4<u32>, n: u32) -> vec4<u32> {
    return (x << vec4<u32>(n)) | (x >> vec4<u32>(32u - n));
}

fn store_words(base: u32, words: vec4<u32>) {
    for (var j = 0u; j < 4u; j++) {
        let position = base + j;
        if position >= params.skip && position - params.skip < params.len {
            let word = words[j];
            var value = word;
            if params.raw_words == 0u {
                value = bitcast<u32>(f32(word >> 8u) * (1.0 / 16777216.0));
            }
            dst[position - params.skip] = value;
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.skip + params.len;
    let count = total / 16u + select(0u, 1u, total % 16u != 0u);
    if gid.x >= count { return; }
    let initial_a = vec4<u32>(0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u);
    var initial_d = params.counter_stream;
    initial_d.x += gid.x;
    initial_d.y += select(0u, 1u, initial_d.x < params.counter_stream.x);
    var a = initial_a;
    var b = params.key0;
    var c = params.key1;
    var d = initial_d;
    for (var round = 0u; round < 6u; round++) {
        a += b; d = rotate(d ^ a, 16u);
        c += d; b = rotate(b ^ c, 12u);
        a += b; d = rotate(d ^ a, 8u);
        c += d; b = rotate(b ^ c, 7u);
        b = b.yzwx; c = c.zwxy; d = d.wxyz;
        a += b; d = rotate(d ^ a, 16u);
        c += d; b = rotate(b ^ c, 12u);
        a += b; d = rotate(d ^ a, 8u);
        c += d; b = rotate(b ^ c, 7u);
        b = b.wxyz; c = c.zwxy; d = d.yzwx;
    }
    store_words(gid.x * 16u, a + initial_a);
    store_words(gid.x * 16u + 4u, b + params.key0);
    store_words(gid.x * 16u + 8u, c + params.key1);
    store_words(gid.x * 16u + 12u, d + initial_d);
}
