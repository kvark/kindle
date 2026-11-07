//! RGB resize geometry, matching Pillow Resample.c's bilinear support and 22-bit
//! coefficients. Only geometry is computed here, never image pixels.

pub(super) fn coefficients(width: u32, height: u32, side: usize) -> Vec<u32> {
    // Each destination coordinate has [coefficient offset, first source, count].
    let mut result = vec![0; 2 * side * 3];
    for (axis, source) in [width, height].into_iter().enumerate() {
        let scale = f64::from(source) / side as f64;
        let support = scale.max(1.0);
        for output in 0..side {
            let center = (output as f64 + 0.5) * scale;
            let first = (center - support + 0.5).max(0.0) as u32;
            let end = ((center + support + 0.5) as u32).min(source);
            let weights = (first..end)
                .map(|input| {
                    (1.0 - ((f64::from(input) - center + 0.5) * support.recip()).abs()).max(0.0)
                })
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            let entry = (axis * side + output) * 3;
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
pub(super) fn reference(pixels: &[u8], width: usize, height: usize, side: usize) -> Vec<u8> {
    let filter = coefficients(width as u32, height as u32, side);
    let mut horizontal = vec![0_u8; height * side * 3];
    for y in 0..height {
        for x in 0..side {
            let (offset, first, count) = (filter[3 * x], filter[3 * x + 1], filter[3 * x + 2]);
            for channel in 0..3 {
                let mut sum = 1 << 21;
                for i in 0..count {
                    sum += u32::from(pixels[(y * width + (first + i) as usize) * 3 + channel])
                        * filter[(offset + i) as usize];
                }
                horizontal[(y * side + x) * 3 + channel] = (sum >> 22).min(255) as u8;
            }
        }
    }
    let mut output = vec![0_u8; side * side * 3];
    for y in 0..side {
        let entry = (side + y) * 3;
        let (offset, first, count) = (filter[entry], filter[entry + 1], filter[entry + 2]);
        for x in 0..side {
            for channel in 0..3 {
                let mut sum = 1 << 21;
                for j in 0..count {
                    sum += u32::from(horizontal[((first + j) as usize * side + x) * 3 + channel])
                        * filter[(offset + j) as usize];
                }
                output[(y * side + x) * 3 + channel] = (sum >> 22).min(255) as u8;
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
        let actual = reference(&pixels, width, height, 64);
        assert_eq!(
            format!("{:x}", Sha256::digest(actual)),
            digest,
            "{width}x{height}"
        );
    }
}

#[test]
fn smaller_rgb_matches_pillow_byte_fixtures() {
    use sha2::{Digest, Sha256};
    // Pillow 12.3.0 Image.resize, independent RGB bilinear reference.
    for (side, width, height, digest) in [
        (
            32,
            160,
            210,
            "4b0c7f4a29a26b145f711402cd21f06777aa112255fce516bfa70ef499500dbf",
        ),
        (
            32,
            64,
            64,
            "3cb333e99f19485b07623f0116321ea4423c16b5c8e4ebd77312f6b7bcf32bc7",
        ),
        (
            32,
            13,
            7,
            "29ed015bb3a2bc3c03226362a5d053dc62f8d26944c04c0cce6df8ba9e1404de",
        ),
        (
            32,
            1,
            19,
            "0a774e87a78f99b4af0234cfc136cce5f40be2c71314afdb4b1a2f97066683b5",
        ),
        (
            32,
            640,
            480,
            "49b2d477d537e11291c818f1235cdc69fc50d6851b9c2a5fd70b919ce19b5847",
        ),
        (
            32,
            641,
            479,
            "f38c8ad026e881d310806031da5b6d6015186ef3b68f8435bd3827b738fcd85f",
        ),
        (
            32,
            1,
            1,
            "5fab80bea92e663dde656c6edf6228d7c27b4a1159b7adac8e3575e579217a1d",
        ),
        (
            48,
            160,
            210,
            "ed2ccac5dcfb095a03652127a6c75feebcf1844dd205a5d534d18cb5f2b78b34",
        ),
        (
            48,
            64,
            64,
            "88ef6c1310450217a45f0809a51637c5a5e311363ae150be3342b87c852d709a",
        ),
        (
            48,
            13,
            7,
            "6070eb3d7fcf06f8d5df6a61d498b9b4b0997e32412c0be4c8c00f770181f360",
        ),
        (
            48,
            1,
            19,
            "c09c55443db679016164a4360271ef3b73978c3f5d96469b7757a57d875aaf6d",
        ),
        (
            48,
            640,
            480,
            "32769f9f24abe447c63ae05ceac8e09c6fd5d74b6fc7fe587a45792e91375a71",
        ),
        (
            48,
            641,
            479,
            "decf9d5f8d08d86c0232a7a3b8d97d1c2cc0b01742d6f5fe75f4d6700a2ec652",
        ),
        (
            48,
            1,
            1,
            "e5196e93a00fd12eb0259e84f6e4345c85bf38784c188ca12b4107d9e36bc58b",
        ),
    ] {
        let pixels = (0..width * height * 3)
            .map(|i| ((i * 37 + i / 43) % 256) as u8)
            .collect::<Vec<_>>();
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(reference(&pixels, width, height, side))
            ),
            digest,
            "{width}x{height} to {side}"
        );
    }
}
