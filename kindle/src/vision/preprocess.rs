// Adapted from kvark/dinovision at dc35cdf1c7c910cdd93c5b5362846842ae469a21 (MIT).
//! Turning an image into the `"patches"` tensor the encoder graph wants.
//!
//! The graph folds DINOv3's patch-embedding Conv2d into a single matmul,
//! which means the flattening order here has to agree exactly with the
//! order the convolution weight was flattened in. PyTorch stores that
//! weight as `[out_channels, in_channels, kh, kw]`, so within one patch
//! the element order is **channel-major**:
//!
//! ```text
//! index = c * patch_size² + ky * patch_size + kx
//! ```
//!
//! Get this wrong and the model still runs, producing confident nonsense
//! — so it is pinned down by tests below.

/// ImageNet statistics from the model's `preprocessor_config.json`.
pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Resize an arbitrary interleaved RGB8 frame into a square without
/// distorting its aspect ratio. Unused pixels are filled with the ImageNet
/// mean, which becomes approximately zero after DINO normalization.
pub fn resize_letterbox_rgb8(rgb: &[u8], width: usize, height: usize, target: usize) -> Vec<u8> {
    assert!(width > 0 && height > 0 && target > 0);
    let source_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .expect("source RGB dimensions overflow usize");
    assert_eq!(rgb.len(), source_len);
    if width == target && height == target {
        return rgb.to_vec();
    }

    let scale = (target as f32 / width as f32).min(target as f32 / height as f32);
    let scaled_width = (width as f32 * scale).round().clamp(1.0, target as f32) as usize;
    let scaled_height = (height as f32 * scale).round().clamp(1.0, target as f32) as usize;
    let offset_x = (target - scaled_width) / 2;
    let offset_y = (target - scaled_height) / 2;
    let fill = IMAGE_MEAN.map(|value| (value * 255.0).round() as u8);
    let target_len = target
        .checked_mul(target)
        .and_then(|pixels| pixels.checked_mul(3))
        .expect("target RGB dimensions overflow usize");
    let mut output = vec![0_u8; target_len];
    for pixel in output.as_chunks_mut::<3>().0 {
        pixel.copy_from_slice(&fill);
    }

    for target_y in 0..scaled_height {
        let source_y = ((target_y as f32 + 0.5) * height as f32 / scaled_height as f32 - 0.5)
            .clamp(0.0, (height - 1) as f32);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let mix_y = source_y - y0 as f32;
        for target_x in 0..scaled_width {
            let source_x = ((target_x as f32 + 0.5) * width as f32 / scaled_width as f32 - 0.5)
                .clamp(0.0, (width - 1) as f32);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let mix_x = source_x - x0 as f32;
            let destination = ((target_y + offset_y) * target + target_x + offset_x) * 3;
            for channel in 0..3 {
                let top = rgb[(y0 * width + x0) * 3 + channel] as f32 * (1.0 - mix_x)
                    + rgb[(y0 * width + x1) * 3 + channel] as f32 * mix_x;
                let bottom = rgb[(y1 * width + x0) * 3 + channel] as f32 * (1.0 - mix_x)
                    + rgb[(y1 * width + x1) * 3 + channel] as f32 * mix_x;
                output[destination + channel] =
                    (top * (1.0 - mix_y) + bottom * mix_y).round() as u8;
            }
        }
    }
    output
}

/// Flatten an already-normalized CHW pixel tensor into patches.
///
/// `pixels` is `[3, image_size, image_size]`, matching what HuggingFace's
/// image processor hands to the model as `pixel_values`. Taking this form
/// directly is what lets the desktop verifier feed the exact same tensor
/// as the reference implementation, keeping resize and normalization
/// differences out of a numerics comparison.
///
/// Returns `[num_patches, patch_dim]` in row-major grid order.
pub fn patches_from_pixels_chw(pixels: &[f32], size: usize, ps: usize) -> Vec<f32> {
    assert!(ps > 0 && size > 0 && size.is_multiple_of(ps));
    let grid = size / ps;
    assert_eq!(
        pixels.len(),
        3 * size * size,
        "expected a [3, {size}, {size}] pixel tensor, got {} values",
        pixels.len()
    );

    let plane = size * size;
    let patch_area = ps * ps;
    let patch_dim = 3 * patch_area;
    let mut out = vec![0.0f32; grid * grid * patch_dim];

    for gy in 0..grid {
        for gx in 0..grid {
            let patch = (gy * grid + gx) * patch_dim;
            for c in 0..3 {
                for ky in 0..ps {
                    let src_row = c * plane + (gy * ps + ky) * size + gx * ps;
                    let dst_row = patch + c * patch_area + ky * ps;
                    out[dst_row..dst_row + ps].copy_from_slice(&pixels[src_row..src_row + ps]);
                }
            }
        }
    }

    out
}

/// Flatten interleaved 8-bit RGB into patches, rescaling to `[0, 1]` and
/// applying the ImageNet normalization on the way.
///
/// `rgb` is `[image_size, image_size, 3]` — the layout image decoders and
/// camera conversions naturally produce. No resizing happens here; the
/// caller supplies an image already at `config.image_size`.
pub fn patches_from_rgb8(rgb: &[u8], size: usize, ps: usize) -> Vec<f32> {
    assert!(ps > 0 && size > 0 && size.is_multiple_of(ps));
    let grid = size / ps;
    assert_eq!(
        rgb.len(),
        3 * size * size,
        "expected a [{size}, {size}, 3] RGB image, got {} bytes",
        rgb.len()
    );

    let patch_area = ps * ps;
    let patch_dim = 3 * patch_area;
    let mut out = vec![0.0f32; grid * grid * patch_dim];

    for gy in 0..grid {
        for gx in 0..grid {
            let patch = (gy * grid + gx) * patch_dim;
            for ky in 0..ps {
                let y = gy * ps + ky;
                for kx in 0..ps {
                    let x = gx * ps + kx;
                    let src = (y * size + x) * 3;
                    for c in 0..3 {
                        let v = rgb[src + c] as f32 / 255.0;
                        out[patch + c * patch_area + ky * ps + kx] =
                            (v - IMAGE_MEAN[c]) / IMAGE_STD[c];
                    }
                }
            }
        }
    }

    out
}

/// Reshape a `[out, 3, patch, patch]` Conv2d weight into the
/// `[patch_dim, out]` matrix the graph's patch-embedding matmul expects.
///
/// The source is already contiguous in channel-major order per output
/// channel, so this is purely a transpose of a `[out, patch_dim]` view.
pub fn conv_weight_to_matmul(weight: &[f32], out_channels: usize, patch_dim: usize) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        out_channels * patch_dim,
        "conv weight has {} values, expected {out_channels} * {patch_dim}",
        weight.len()
    );
    let mut m = vec![0.0f32; patch_dim * out_channels];
    for o in 0..out_channels {
        for i in 0..patch_dim {
            m[i * out_channels + o] = weight[o * patch_dim + i];
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::dinov3::Config;

    #[test]
    fn chw_patch_layout_is_channel_major() {
        let c = Config::vits16();
        // Encode each pixel's identity as its flat CHW index so the
        // mapping is checkable by arithmetic.
        let size = c.image_size;
        let pixels: Vec<f32> = (0..3 * size * size).map(|i| i as f32).collect();
        let patches = patches_from_pixels_chw(&pixels, c.image_size, c.patch_size);

        let ps = c.patch_size;
        let plane = size * size;
        // Patch (gy=3, gx=5), channel 2, offset (ky=7, kx=11).
        let (gy, gx, ch, ky, kx) = (3, 5, 2, 7, 11);
        let got = patches[(gy * c.grid() + gx) * c.patch_dim() + ch * ps * ps + ky * ps + kx];
        let want = (ch * plane + (gy * ps + ky) * size + gx * ps + kx) as f32;
        assert_eq!(got, want);
    }

    #[test]
    fn rgb8_and_chw_paths_agree() {
        let c = Config::vits16();
        let size = c.image_size;
        // Build an arbitrary but reproducible RGB image, then the
        // equivalent normalized CHW tensor, and check both flatteners
        // land on the same patch tensor.
        let rgb: Vec<u8> = (0..3 * size * size).map(|i| (i % 251) as u8).collect();
        let mut chw = vec![0.0f32; 3 * size * size];
        for y in 0..size {
            for x in 0..size {
                for ch in 0..3 {
                    let v = rgb[(y * size + x) * 3 + ch] as f32 / 255.0;
                    chw[ch * size * size + y * size + x] = (v - IMAGE_MEAN[ch]) / IMAGE_STD[ch];
                }
            }
        }

        let from_rgb = patches_from_rgb8(&rgb, c.image_size, c.patch_size);
        let from_chw = patches_from_pixels_chw(&chw, c.image_size, c.patch_size);
        assert_eq!(from_rgb.len(), from_chw.len());
        let worst = from_rgb
            .iter()
            .zip(&from_chw)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-6, "paths disagree by {worst}");
    }

    #[test]
    fn normalization_maps_midgray_near_zero() {
        let c = Config::vits16();
        // 0.485*255 ≈ 124 is the red-channel mean, so red lands near 0.
        let rgb = vec![124u8; 3 * c.image_size * c.image_size];
        let patches = patches_from_rgb8(&rgb, c.image_size, c.patch_size);
        assert!(
            patches[0].abs() < 0.02,
            "red channel not centred: {}",
            patches[0]
        );
    }

    #[test]
    fn conv_weight_transpose_roundtrip() {
        // [out=2, patch_dim=3] stored row-major becomes [3, 2].
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = conv_weight_to_matmul(&w, 2, 3);
        assert_eq!(m, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn letterbox_preserves_aspect_ratio_and_uses_mean_padding() {
        let input = vec![255_u8, 0, 0, 0, 255, 0];
        let output = resize_letterbox_rgb8(&input, 2, 1, 4);
        assert_eq!(output.len(), 4 * 4 * 3);
        let fill = IMAGE_MEAN.map(|value| (value * 255.0).round() as u8);
        assert_eq!(&output[..3], &fill);
        assert_eq!(&output[3 * 4 * 3..3 * 4 * 3 + 3], &fill);
        assert_ne!(&output[4 * 3..4 * 3 + 3], &fill);
    }
}
