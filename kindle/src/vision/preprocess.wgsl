struct Params {
    image: vec4<u32>, // width, height, byte stride, channels
    source: vec4<u32>, // byte offset, BGR flag, output size, patch size
    region: vec4<u32>, // resized width/height, letterbox x/y
    output: vec4<u32>, // stream output offset
}

var<storage, read> pixels: array<u32>;
var<storage, read_write> patches: array<f32>;
var<uniform> params: Params;

fn channel(x: u32, y: u32, c: u32) -> f32 {
    let component = select(c, 2u - c, params.source.y != 0u);
    let byte = params.source.x + y * params.image.z + x * params.image.w + component;
    return f32((pixels[byte / 4u] >> ((byte % 4u) * 8u)) & 255u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.source.z;
    if id.x >= size * size { return; }
    let x = id.x % size;
    let y = id.x / size;
    let ps = params.source.w;
    let grid = size / ps;
    let patch_index = (y / ps) * grid + x / ps;
    let local = (y % ps) * ps + x % ps;
    let means = vec3<f32>(0.485, 0.456, 0.406);
    let stds = vec3<f32>(0.229, 0.224, 0.225);
    let inside = x >= params.region.z && x < params.region.z + params.region.x
        && y >= params.region.w && y < params.region.w + params.region.y;
    for (var c = 0u; c < 3u; c += 1u) {
        var normalized = 0.0;
        if inside {
            let sx = clamp((f32(x - params.region.z) + 0.5) * f32(params.image.x)
                / f32(params.region.x) - 0.5, 0.0, f32(params.image.x - 1u));
            let sy = clamp((f32(y - params.region.w) + 0.5) * f32(params.image.y)
                / f32(params.region.y) - 0.5, 0.0, f32(params.image.y - 1u));
            let x0 = u32(floor(sx));
            let y0 = u32(floor(sy));
            let x1 = min(x0 + 1u, params.image.x - 1u);
            let y1 = min(y0 + 1u, params.image.y - 1u);
            let mx = sx - f32(x0);
            let my = sy - f32(y0);
            let top = channel(x0, y0, c) * (1.0 - mx) + channel(x1, y0, c) * mx;
            let bottom = channel(x0, y1, c) * (1.0 - mx) + channel(x1, y1, c) * mx;
            let value = top * (1.0 - my) + bottom * my;
            normalized = (value / 255.0 - means[c]) / stds[c];
        }
        let destination = params.output.x + patch_index * (3u * ps * ps) + c * ps * ps + local;
        patches[destination] = normalized;
    }
}
