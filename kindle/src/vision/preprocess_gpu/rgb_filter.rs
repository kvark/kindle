//! RGB64 geometry, matching Pillow Resample.c's bilinear support and 22-bit
//! coefficients. Only geometry is computed here, never image pixels.

const SIDE: usize = 64;

pub(super) fn coefficients(width: u32, height: u32) -> Vec<u32> {
    // Each destination coordinate has [coefficient offset, first source, count].
    let mut result = vec![0; 2 * SIDE * 3];
    for (axis, source) in [width, height].into_iter().enumerate() {
        let scale = f64::from(source) / SIDE as f64;
        let support = scale.max(1.0);
        for output in 0..SIDE {
            let center = (output as f64 + 0.5) * scale;
            let first = (center - support + 0.5).max(0.0) as u32;
            let end = ((center + support + 0.5) as u32).min(source);
            let weights = (first..end)
                .map(|input| {
                    (1.0 - ((f64::from(input) - center + 0.5) * support.recip()).abs()).max(0.0)
                })
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            let entry = (axis * SIDE + output) * 3;
            let offset = result.len() as u32;
            result[entry..entry + 3].copy_from_slice(&[offset, first, end - first]);
            result.extend(
                weights
                    .into_iter()
                    .map(|weight| (weight / sum * f64::from(1 << 22) + 0.5) as u32),
            );
        }
    }
    result
}

#[cfg(test)]
pub(super) fn reference(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let filter = coefficients(width as u32, height as u32);
    let mut horizontal = vec![0_u8; height * SIDE * 3];
    for y in 0..height {
        for x in 0..SIDE {
            let (offset, first, count) = (filter[3 * x], filter[3 * x + 1], filter[3 * x + 2]);
            for channel in 0..3 {
                let mut sum = 1 << 21;
                for i in 0..count {
                    sum += u32::from(pixels[(y * width + (first + i) as usize) * 3 + channel])
                        * filter[(offset + i) as usize];
                }
                horizontal[(y * SIDE + x) * 3 + channel] = (sum >> 22).min(255) as u8;
            }
        }
    }
    let mut output = vec![0_u8; SIDE * SIDE * 3];
    for y in 0..SIDE {
        let entry = (SIDE + y) * 3;
        let (offset, first, count) = (filter[entry], filter[entry + 1], filter[entry + 2]);
        for x in 0..SIDE {
            for channel in 0..3 {
                let mut sum = 1 << 21;
                for j in 0..count {
                    sum += u32::from(horizontal[((first + j) as usize * SIDE + x) * 3 + channel])
                        * filter[(offset + j) as usize];
                }
                output[(y * SIDE + x) * 3 + channel] = (sum >> 22).min(255) as u8;
            }
        }
    }
    output
}

#[test]
fn rgb64_matches_pillow_byte_fixtures() {
    use sha2::{Digest, Sha256};

    // Pillow 12.3.0 Image.resize((64, 64), Image.Resampling.BILINEAR), RGB.
    for (width, height, digest) in [
        (
            160,
            210,
            "88a1e00849d092f3f412a29433a72af590455215edeba0dd3a01eb3466efdbdc",
        ),
        (
            64,
            64,
            "9db8fade220c258d98cad173aa90ff00b57cb896fd36137c219245e6951517d0",
        ),
        (
            13,
            7,
            "12e0ec79357586674091d298d2da51c506ee1a145fa5ea085f2d33e06bad8f89",
        ),
        (
            1,
            19,
            "be0f4c40429fad545db320261acace9a210c76b22a328577ecc05f1603bd059d",
        ),
        (
            640,
            480,
            "955458ac00cb4b1d3492d51d08c1439ee798b0987308f70b5b2f205938145f2a",
        ),
        (
            641,
            479,
            "ba228e43f7dfe2c4eebebf161710ecc8ea63bb8874536dbd227d1508e8937a4f",
        ),
        (
            1,
            1,
            "61a1e4c70dd5a6aef19dac367426fdf238f124c78833998f4402e6643cd2d3e5",
        ),
    ] {
        let pixels = (0..width * height * 3)
            .map(|i| ((i * 37 + i / 43) % 256) as u8)
            .collect::<Vec<_>>();
        let actual = reference(&pixels, width, height);
        assert_eq!(
            format!("{:x}", Sha256::digest(actual)),
            digest,
            "{width}x{height}"
        );
    }
}
