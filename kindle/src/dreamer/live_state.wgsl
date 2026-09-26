struct Params { rows: u32, deter: u32, stoch: u32, unused: u32 }
var<storage, read> next_deter: array<f32>;
var<storage, read> next_stoch: array<f32>;
var<storage, read> arrivals: array<u32>;
var<storage, read_write> previous_deter: array<f32>;
var<storage, read_write> previous_stoch: array<f32>;
var<storage, read_write> feature: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.deter + params.stoch;
    let row = id.x / width;
    let col = id.x % width;
    if row >= params.rows || arrivals[row] == 0u { return; }
    var value: f32;
    if col < params.deter {
        value = next_deter[row * params.deter + col];
        previous_deter[row * params.deter + col] = value;
    } else {
        let index = row * params.stoch + col - params.deter;
        value = next_stoch[index];
        previous_stoch[index] = value;
    }
    feature[id.x] = value;
}
