use super::*;

#[test]
fn typed_staging_preserves_storage_contract() {
    use meganeura::{
        DType,
        codegen::{MatrixTypes, ShaderGroup, TypedMatmulLayout},
    };
    let name = |ty| match ty {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => unreachable!(),
    };
    for storage in [DType::F32, DType::F16, DType::BF16] {
        for output in [DType::F32, DType::BF16] {
            let types = MatrixTypes {
                output,
                ..MatrixTypes::reduced(storage)
            };
            for group in [
                ShaderGroup::MatMul,
                ShaderGroup::MatMulAT,
                ShaderGroup::MatMulBT,
            ] {
                let layout = TypedMatmulLayout::for_product(types, false, [32, 256, 18]);
                let module = meganeura::codegen::generate_module_coop_typed_with_layout(
                    group,
                    types,
                    [16, 16, 16],
                    layout,
                );
                let input_name = name(storage);
                assert!(
                    module
                        .source
                        .contains(&format!("matrix_a: array<vec4<{input_name}>>"))
                );
                assert!(
                    module
                        .source
                        .contains(&format!("matrix_b: array<vec4<{input_name}>>"))
                );
                assert!(
                    module
                        .source
                        .contains(&format!("matrix_c: array<{}>", name(output)))
                );
            }
        }
    }
}

fn summarize(root: &Path, name: &str, values: &[f32]) {
    let invalid = values
        .iter()
        .enumerate()
        .filter_map(|(i, value)| (!value.is_finite()).then_some(i))
        .collect::<Vec<_>>();
    save_json(
        &root.join(format!("{name}.json")),
        &serde_json::json!({
            "elements": values.len(), "invalid": invalid.len(),
            "first_invalid": invalid.first(),
            "maximum_finite_absolute": values.iter().filter(|v| v.is_finite()).fold(0.0_f32, |a, b| a.max(b.abs())),
            "bits": values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        }),
    );
    eprintln!(
        "{name}: {} values, {} nonfinite",
        values.len(),
        invalid.len()
    );
}

#[test]
#[ignore = "bounded first-value diagnosis, no learner updates"]
fn diagnose() {
    let source = std::env::var_os("KINDLE_DREAMER_PROFILE_SOURCE").unwrap();
    let root = std::path::PathBuf::from(std::env::var_os("KINDLE_DREAMER_PROFILE_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let mut core = restore_profile_core(Path::new(&source));
    assert_eq!(
        core.gpu_device().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!core.gpu_device().is_software_emulated);
    let budget = core.gpu_memory_budget();
    assert!(budget.budget_bytes.saturating_sub(budget.usage_bytes) >= 2 << 30);
    checkpoint(&mut core, &root.join("initial"));
    // Exact sparse input probes distinguish dense-kernel data contracts from
    // model graph composition. Both checked and staged shapes are present.
    let mut cases = Vec::new();
    for (m, n, k) in [(32, 256, 18), (32, 256, 256), (512, 256, 256)] {
        for transpose in 0..3 {
            for rounded in [false, true] {
                let mut graph = meganeura::Graph::new();
                let a_shape = if transpose == 1 { [k, m] } else { [m, k] };
                let b_shape = if transpose == 2 { [n, k] } else { [k, n] };
                let a = graph.parameter("a", &a_shape);
                let b = graph.parameter("b", &b_shape);
                let a = graph.to_bf16(a);
                let b = graph.to_bf16(b);
                let output = match transpose {
                    0 => graph.matmul(a, b),
                    1 => graph.matmul_at(a, b),
                    _ => graph.matmul_bt(a, b),
                };
                let output = if rounded {
                    let output = graph.to_bf16(output);
                    graph.from_bf16(output)
                } else {
                    output
                };
                graph.set_outputs(vec![output]);
                let mut a = vec![0.0; m * k];
                let b = (0..k * n)
                    .map(|i| (i * 31 % 61 + 1) as f32 / 4.0)
                    .collect::<Vec<_>>();
                let mut expected = vec![0.0; m * n];
                for row in 0..m {
                    let inner = (row * 137 + k - 1) % k;
                    a[if transpose == 1 {
                        inner * m + row
                    } else {
                        row * k + inner
                    }] = 1.0;
                    for col in 0..n {
                        expected[row * n + col] = b[if transpose == 2 {
                            col * k + inner
                        } else {
                            inner * n + col
                        }];
                    }
                }
                let mut session = meganeura::Session::with_context_opts(
                    meganeura::compile::compile(&graph),
                    core.gpu.clone(),
                    meganeura::SessionOptions {
                        poison: true,
                        ..Default::default()
                    },
                );
                session.set_parameter("a", &a);
                session.set_parameter("b", &b);
                session.step();
                session.wait();
                let actual = session.read_output(m * n);
                let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                cases.push(serde_json::json!({"m":m,"n":n,"k":k,"transpose":transpose,"rounded":rounded,
                    "mismatches":mismatches,"elements":actual.len(),"pipelines":session.dispatch_pipeline_keys()}));
                save_json(
                    &root.join(format!("dense-{}.json", cases.len())),
                    cases.last().unwrap(),
                );
                assert_eq!(mismatches, 0, "dense case {}", cases.len());
            }
        }
    }
    let mut rng = StdRng::seed_from_u64(701);
    let batch = fixture(&core.config, &mut rng);
    let posterior = core.sample_posterior_batch(&batch);
    summarize(&root, "posterior-deter", &flatten_time(&posterior.deter));
    summarize(&root, "posterior-stoch", &flatten_time(&posterior.stoch));
    let mut targets = core.imagine_and_target(&batch, &posterior);
    core.imagination.wait();
    core.behavior_slow.wait();
    for (name, session) in [
        ("imagination", &core.imagination),
        ("slow", &core.behavior_slow),
    ] {
        for index in 0..session.num_outputs() {
            let elements = session.slot_size(ExternalSlot::Output(index)).unwrap() / 4;
            let mut values = vec![0.0; elements];
            session.read_output_by_index(index, &mut values);
            summarize(&root, &format!("{name}-{index}"), &values);
        }
    }
    core.materialize_targets(&mut targets);
    for (name, values) in [
        ("target-action", &targets.action_target),
        ("target-value", &targets.imagined_value_target),
        ("target-weight", &targets.imagined_weight),
    ] {
        summarize(&root, name, values);
    }
    checkpoint(&mut core, &root.join("final"));
    save_json(
        &root.join("result.json"),
        &serde_json::json!({
            "status":"complete", "learner_updates":0,"game_actions":0,"dense_cases":cases.len(),
        }),
    );
}
