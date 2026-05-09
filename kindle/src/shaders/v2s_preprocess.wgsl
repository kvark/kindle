// V2-S preprocess: BGRA8 (or RGBA8) source frame → RGB f32 192×192, written
// into one lane's slice of the V2-S `image` input buffer.
//
// Producer:
//   `src_words` — packed `array<u32>` view of one Dullahan-exported frame,
//   tightly packed BGRA8 (or RGBA8) at `params.src_w * params.src_h` pixels,
//   with the first u32 at the slot's frame_data offset (the host wraps the
//   imported buffer with a `BufferPiece { offset = data_offset + slot_idx *
//   bytes_per_frame }` so this shader sees pixel 0 at index 0).
//
// Consumer:
//   `dst` — flat `array<f32>` view of V2-S's `image` input slot, laid out
//   `[batch=N, channels=3, h=192, w=192]` row-major (matches what
//   `stage_image_to_kindle` writes today).
//
// Each workgroup invocation writes the three colour channels of one output
// pixel for `params.lane`.  Dispatch `(192/16, 192/16, 1)` workgroups per
// lane; the host issues one dispatch per lane per step.
//
// Sampling: bilinear, with `+0.5 px` half-pixel centring matching
// `F.interpolate(mode='bilinear', align_corners=False)` (the convention
// `stage_image_to_kindle` used in the host CUDA path).

struct Params {
    /// 0 = BGRA byte order in memory (DXVK / VK_FORMAT_B8G8R8A8_UNORM)
    /// 1 = RGBA byte order (VK_FORMAT_R8G8B8A8_UNORM)
    is_rgba: u32,
    /// Lane index (0..N).  Selects the destination slice in `dst`.
    lane: u32,
    src_w: u32,
    src_h: u32,
    /// Always 192 today, kept as a uniform so a future config change
    /// doesn't require a shader recompile.
    dst_hw: u32,
    /// Stride (in u32s, == bytes/4) between rows of the source frame.
    /// For tightly-packed BGRA8 this is just src_w; left explicit so a
    /// padded layout can plug in later.
    src_row_stride: u32,
    /// Index into `src_words` of the first pixel of the current ring
    /// slot.  Computed host-side as
    ///     (frame_data_offset_bytes + slot_idx * bytes_per_frame) / 4
    /// so the shader doesn't need to know the SHM header layout or
    /// which slot the producer just wrote.  Bumped per dispatch.
    src_frame_words_offset: u32,
    /// Number of active HUD-mask rects in `hud_rects` (0..MAX_HUD_RECTS).
    /// Mask rects are in *destination-space* (post-resize) pixel
    /// coordinates: any output pixel inside one is forced to all-zero
    /// before being written to V2-S's image buffer.  Lets the agent see
    /// only game-world pixels and not learn HP-bar / ammo-counter
    /// gradients via the frozen ImageNet backbone.
    hud_count: u32,
}

/// Capacity of the inline mask array.  4 covers every game we care
/// about today (PW: 1 HP-bar rect; the rest empty); bumping this
/// requires re-issuing the shader + matching the Rust `PreprocessParams`
/// struct.
const MAX_HUD_RECTS: u32 = 4u;
struct HudRects {
    /// Each `vec4<u32>` is `(x0, y0, x1, y1)` in dst-space (0..dst_hw).
    /// Pixel `(x, y)` is masked when `x0 <= x < x1 && y0 <= y < y1`.
    rects: array<vec4<u32>, 4>,
}

var<storage, read> src_words: array<u32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<uniform> hud_rects: HudRects;

fn fetch_rgb(px: u32, py: u32) -> vec3<f32> {
    let word = src_words[params.src_frame_words_offset
        + py * params.src_row_stride + px];
    let b = f32((word) & 0xFFu) * (1.0 / 255.0);
    let g = f32((word >> 8u) & 0xFFu) * (1.0 / 255.0);
    let r = f32((word >> 16u) & 0xFFu) * (1.0 / 255.0);
    // The 4th byte is alpha — discarded.
    if params.is_rgba == 1u {
        // Memory order is R-G-B-A: reinterpret byte 0 as R, byte 2 as B.
        return vec3<f32>(b, g, r);
    }
    return vec3<f32>(r, g, b);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_x = gid.x;
    let out_y = gid.y;
    if out_x >= params.dst_hw || out_y >= params.dst_hw {
        return;
    }

    // Bilinear source coordinate, half-pixel centred (align_corners=False).
    let scale_x = f32(params.src_w) / f32(params.dst_hw);
    let scale_y = f32(params.src_h) / f32(params.dst_hw);
    let sx = (f32(out_x) + 0.5) * scale_x - 0.5;
    let sy = (f32(out_y) + 0.5) * scale_y - 0.5;

    let x0_f = floor(sx);
    let y0_f = floor(sy);
    let fx = sx - x0_f;
    let fy = sy - y0_f;

    let max_x = i32(params.src_w) - 1;
    let max_y = i32(params.src_h) - 1;
    let x0 = u32(clamp(i32(x0_f), 0, max_x));
    let y0 = u32(clamp(i32(y0_f), 0, max_y));
    let x1 = u32(clamp(i32(x0_f) + 1, 0, max_x));
    let y1 = u32(clamp(i32(y0_f) + 1, 0, max_y));

    let c00 = fetch_rgb(x0, y0);
    let c10 = fetch_rgb(x1, y0);
    let c01 = fetch_rgb(x0, y1);
    let c11 = fetch_rgb(x1, y1);
    let cx0 = mix(c00, c10, fx);
    let cx1 = mix(c01, c11, fx);
    var rgb = mix(cx0, cx1, fy);

    // HUD masking: zero out any pixel in a mask rect.  Done after
    // sampling so the bilinear taps don't bleed mask-zero-pixels into
    // their neighbors — same convention as the host CUDA path's
    // post-interpolate `t[..., my0:my1, mx0:mx1] = 0`.
    let n_hud = min(params.hud_count, MAX_HUD_RECTS);
    for (var i = 0u; i < n_hud; i = i + 1u) {
        let r = hud_rects.rects[i];
        if out_x >= r.x && out_x < r.z && out_y >= r.y && out_y < r.w {
            rgb = vec3<f32>(0.0, 0.0, 0.0);
            break;
        }
    }

    // Write into [lane, c, out_y, out_x] for c in 0..3.
    let plane = params.dst_hw * params.dst_hw;        // 192*192
    let lane_stride = 3u * plane;                      // 192*192*3
    let lane_base = params.lane * lane_stride;
    let pixel = out_y * params.dst_hw + out_x;
    dst[lane_base + 0u * plane + pixel] = rgb.x;
    dst[lane_base + 1u * plane + pixel] = rgb.y;
    dst[lane_base + 2u * plane + pixel] = rgb.z;
}
