use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

mod qualification;

#[test]
fn shaders_validate_offline() {
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_TARGET_SHADER_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let mut sources = vec![
        (
            "returns".to_owned(),
            include_str!("../targets_returns.wgsl").to_owned(),
            ReturnsData::layout(),
        ),
        (
            "encode".to_owned(),
            include_str!("../targets_encode.wgsl").to_owned(),
            EncodeData::layout(),
        ),
        (
            "actor".to_owned(),
            include_str!("../targets_actor.wgsl").to_owned(),
            ActorData::layout(),
        ),
        (
            "report".to_owned(),
            include_str!("../targets_report.wgsl").to_owned(),
            ReportData::layout(),
        ),
    ];
    for count in [2, 8, 32, 256, 2048, 8192] {
        sources.push((
            format!("normalize-{count}"),
            include_str!("../targets_normalize.wgsl").replace("_SORT_LENGTH_", &count.to_string()),
            NormalizeData::layout(),
        ));
    }
    let mut entries = 0;
    for (name, source, layout) in sources {
        std::fs::write(root.join(format!("{name}.wgsl")), &source).unwrap();
        let mut module = naga::front::wgsl::parse_str(&source).unwrap();
        for (_, global) in module.global_variables.iter_mut() {
            if matches!(
                global.space,
                naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
            ) {
                let binding = layout
                    .bindings
                    .iter()
                    .position(|(name, _)| Some(*name) == global.name.as_deref())
                    .unwrap();
                global.binding = Some(naga::ResourceBinding {
                    group: 0,
                    binding: binding as u32,
                });
            }
        }
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap();
        let mut sizes = naga::proc::Layouter::default();
        sizes.update(module.to_ctx()).unwrap();
        let shared: u32 = module
            .global_variables
            .iter()
            .filter(|(_, v)| v.space == naga::AddressSpace::WorkGroup)
            .map(|(_, v)| sizes[v.ty].size)
            .sum();
        assert!(shared <= 32768);
        for entry in &module.entry_points {
            assert!(matches!(entry.workgroup_size, [64, 1, 1] | [256, 1, 1]));
            entries += 1;
        }
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &naga::back::spv::Options {
                lang_version: (1, 3),
                flags: naga::back::spv::WriterFlags::empty(),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        std::fs::write(root.join(format!("{name}.spv")), bytes).unwrap();
    }
    assert_eq!(entries, 11);
}

#[test]
fn bitonic_schedule_matches_total_order_and_percentile_boundaries() {
    let mut rng = StdRng::seed_from_u64(701);
    for count in [2usize, 3, 6, 31, 127, 512, 1920, 7680] {
        let values: Vec<f32> = (0..count)
            .map(|i| match i % 11 {
                0 => -0.0,
                1 => 0.0,
                2 => -0.125,
                3 => 0.125,
                _ => rng.random_range(-200.0..200.0),
            })
            .collect();
        let mut expected = values.clone();
        expected.sort_by(f32::total_cmp);
        let mut actual = values;
        actual.resize(count.next_power_of_two(), f32::INFINITY);
        let key = |value: f32| {
            let bits = value.to_bits();
            if bits & 0x8000_0000 != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000
            }
        };
        let mut width = 2;
        while width <= actual.len() {
            let mut distance = width / 2;
            while distance > 0 {
                for lane in 0..256 {
                    for i in (lane..actual.len()).step_by(256) {
                        let other = i ^ distance;
                        if other > i && (key(actual[i]) > key(actual[other])) == (i & width == 0) {
                            actual.swap(i, other);
                        }
                    }
                }
                distance /= 2;
            }
            width *= 2;
        }
        assert!(
            actual[..count]
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        for q in [0.05f32, 0.95] {
            let position = q * (count - 1) as f32;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            assert!(lower < count && upper < count);
            let mix = position - lower as f32;
            let got = actual[lower] * (1.0 - mix) + actual[upper] * mix;
            let want = expected[lower] * (1.0 - mix) + expected[upper] * mix;
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }
}
