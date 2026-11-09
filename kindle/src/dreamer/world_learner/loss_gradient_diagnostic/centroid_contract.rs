//! Frozen position-head readback and joint-objective gradient accounting.

use super::*;

const MODES: [&str; 5] = ["rgb", "observed_xy", "future_xy", "kl_events", "joint"];

fn component(g: &mut Graph, c: &DreamerConfig, mode: &str) -> NodeId {
    let o = g.outputs().to_vec();
    let s = c.loss_scales;
    match mode {
        "rgb" => {
            let pair = g.add(o[world::LOSS_RECONSTRUCTION], o[world::LOSS_WORLD_ROLLOUT]);
            g.scale(pair, s.reconstruction)
        }
        "observed_xy" => g.scale(o[19], c.color_centroid_supervision.as_ref().unwrap().scale),
        "future_xy" => g.scale(
            o[world::LOSS_WORLD_CENTROID],
            c.color_centroid_supervision.as_ref().unwrap().scale,
        ),
        "kl_events" => {
            let mut value = g.scale(o[world::LOSS_DYNAMICS], s.dynamics);
            for (index, scale) in [
                (world::LOSS_REPRESENTATION, s.representation),
                (world::LOSS_REWARD, s.reward),
                (world::LOSS_CONTINUATION, s.continuation),
            ] {
                let next = g.scale(o[index], scale);
                value = g.add(value, next);
            }
            value
        }
        "joint" => o[0],
        _ => panic!("unknown component"),
    }
}

fn append_readout(g: &mut Graph, c: &DreamerConfig) {
    let mut out = g.outputs().to_vec();
    assert_eq!(out.len(), 41);
    let parameter = |name: &str| {
        g.nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Parameter { name: p } if p == name))
            .unwrap()
            .id
    };
    let weight = parameter("world.color_centroid.out.weight");
    let bias = parameter("world.color_centroid.out.bias");
    let feature = super::super::super::networks::feature(
        g,
        out[39],
        out[40],
        c.batch_size * c.batch_length,
        c,
    );
    let value = g.matmul(feature, weight);
    out.push(g.bias_add(value, bias));
    g.set_outputs(out);
}

#[test]
fn centroid_component_sum_and_readback_preserve_joint_derivatives() {
    use meganeura::reference::{Feeds, evaluate};
    let mut c = DreamerConfig::tiny(18);
    c.loss_scales.reconstruction = 7.;
    c.loss_scales.dynamics = 0.6;
    c.loss_scales.representation = 0.2;
    c.loss_scales.reward = 2.;
    c.loss_scales.continuation = 3.;
    c.color_centroid_supervision = Some(super::super::super::ColorCentroidSupervision {
        scale: 123.,
        channel: 0,
        background_scale: 1.5,
        region: [0., 0., 1., 1.],
    });
    let mut g = Graph::new();
    let mut out = (0..41)
        .map(|i| g.parameter(&format!("p{i}"), &[1]))
        .collect::<Vec<_>>();
    let mut total = g.scalar(0.);
    for (i, s) in [
        (1, 7.),
        (25, 7.),
        (2, 0.6),
        (3, 0.2),
        (4, 2.),
        (5, 3.),
        (19, 123.),
        (37, 123.),
    ] {
        let value = g.scale(out[i], s);
        total = g.add(total, value);
    }
    out[0] = total;
    g.set_outputs(out);
    let mut feeds = Feeds::new();
    for i in 0..41 {
        feeds.set(&format!("p{i}"), &[0.1 + i as f32 / 17.]);
    }
    let values = MODES
        .iter()
        .map(|mode| {
            let mut selected = g.deep_clone();
            let value = component(&mut selected, &c, mode);
            selected.set_outputs(vec![value]);
            let derivative = meganeura::autodiff::differentiate(&selected);
            let r = evaluate(&derivative, &feeds).unwrap();
            derivative
                .outputs()
                .iter()
                .map(|&n| r[n as usize].data[0])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for i in 0..values[0].len() {
        assert!((values[..4].iter().map(|v| v[i]).sum::<f64>() - values[4][i]).abs() < 1e-5);
    }
}

#[test]
#[ignore = "requires declared guarded frozen centroid contract measurement"]
fn frozen_centroid_contract() {
    let root = PathBuf::from(std::env::var("KINDLE_CENTROID_CONTRACT_ROOT").unwrap());
    let declaration: Value =
        serde_json::from_slice(&fs::read(root.join("execution-pins.json")).unwrap()).unwrap();
    for (path, hash) in declaration["pins"].as_object().unwrap() {
        assert_eq!(
            crate::vision::checkpoint_sha256(Path::new(path)).unwrap(),
            hash.as_str().unwrap()
        );
    }
    let settings: Value =
        serde_json::from_slice(&fs::read(root.join("settings.json")).unwrap()).unwrap();
    let c: DreamerConfig = serde_json::from_value(settings["config"].clone()).unwrap();
    c.validate();
    validate_config(&c).unwrap();
    assert_eq!(
        (c.batch_size, c.batch_length, c.world_backprop_length),
        (8, 32, 32)
    );
    assert_eq!(c.color_centroid_supervision.as_ref().unwrap().scale, 1000.);
    let checkpoint = PathBuf::from(settings["checkpoint"].as_str().unwrap());
    let hash = crate::vision::checkpoint_sha256(&checkpoint).unwrap();
    let anchor = SafeTensorsModel::load(checkpoint.clone()).unwrap();
    assert_eq!(anchor.tensor_info().len(), 210);
    let out = root.join("native-results");
    fs::create_dir(&out).unwrap();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let started = Instant::now();
    // Check the actual native affine operator on saved inference features.
    let features = SafeTensorsModel::load(root.join("inference-features.safetensors")).unwrap();
    let mut graph = Graph::new();
    let input = graph.input("feature", &[128, 2560]);
    let w = graph.parameter("world.color_centroid.out.weight", &[2560, 2]);
    let b = graph.parameter("world.color_centroid.out.bias", &[2]);
    let value = graph.matmul(input, w);
    let value = graph.bias_add(value, b);
    graph.set_outputs(vec![value]);
    let mut session = runtime::build_learner_inference_session(&graph, &gpu, &c);
    for name in [
        "world.color_centroid.out.weight",
        "world.color_centroid.out.bias",
    ] {
        session.set_parameter(name, &anchor.tensor_f32(name).unwrap());
    }
    for name in ["observed", "prior1", "prior8"] {
        session.set_input("feature", &features.tensor_f32(name).unwrap());
        session.step();
        session.wait();
        let mut values = vec![0.; 256];
        session.read_output_by_index(0, &mut values);
        assert!(values.iter().all(|v| v.is_finite()));
        fs::write(
            out.join(format!("inference-{name}.bin")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    drop(session);
    let mut log = BufWriter::new(File::create_new(out.join("steps.jsonl")).unwrap());
    let mut count = 0;
    for mode in MODES {
        let mut graph =
            world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionCrossEntropy);
        append_readout(&mut graph, &c);
        let loss = component(&mut graph, &c, mode);
        let mut outputs = graph.outputs().to_vec();
        outputs[0] = loss;
        graph.set_outputs(outputs);
        let mut parameters = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();
        parameters.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(parameters.len(), 70);
        let mut offset = 0;
        let layout = parameters
            .iter()
            .map(|(name, shape)| {
                let length = shape.iter().product::<usize>();
                let v = json!({"name":name,"shape":shape,"length":length,"offset_f32":offset});
                offset += length;
                v
            })
            .collect::<Vec<_>>();
        assert_eq!(offset, 8_857_797);
        fs::write(
            out.join(format!("layout-{mode}.json")),
            serde_json::to_vec_pretty(&layout).unwrap(),
        )
        .unwrap();
        let mut session = runtime::build_training_session(&graph, &gpu, &c);
        session.clear_optimizer();
        for (name, _) in &parameters {
            session.set_parameter(name, &anchor.tensor_f32(name).unwrap());
        }
        for batch_spec in settings["batches"].as_array().unwrap() {
            let batch = load_batch(batch_spec["path"].as_str().unwrap());
            for (draw, seeds) in settings["seeds"].as_array().unwrap().iter().enumerate() {
                let feeds = inputs(
                    &c,
                    &batch,
                    16,
                    [seeds[0].as_u64().unwrap(), seeds[1].as_u64().unwrap()],
                );
                validate_feeds(&graph, &feeds);
                let input_hash = hash_inputs(&feeds);
                for (name, values) in feeds {
                    session.set_input(&name, &values);
                }
                session.step();
                session.wait();
                assert_eq!(session.adam_step_count(), 0);
                let mut losses = Vec::new();
                for index in [0, 1, 2, 3, 4, 5, 19, 20, 25, 26, 37, 38] {
                    let mut v = [0.];
                    session.read_output_by_index(index, &mut v);
                    assert!(v[0].is_finite());
                    losses.push(v[0]);
                }
                let prefix = format!("{}-{draw}", batch_spec["name"].as_str().unwrap());
                let mut file = BufWriter::new(
                    File::create_new(out.join(format!("gradient-{mode}-{prefix}.bin"))).unwrap(),
                );
                for (name, _) in &parameters {
                    let mut v = vec![0.; session.param_size(name).unwrap()];
                    if session.has_param_grad(name) {
                        session.read_param_grad(name, &mut v);
                    }
                    assert!(v.iter().all(|x| x.is_finite()));
                    file.write_all(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                        .unwrap();
                }
                file.flush().unwrap();
                if mode == "joint" {
                    for (index, name, length) in [
                        (39, "h", 256 * c.network().deter),
                        (40, "z", 256 * c.network().stoch * c.network().classes),
                        (41, "xy", 512),
                    ] {
                        let mut v = vec![0.; length];
                        session.read_output_by_index(index, &mut v);
                        assert!(v.iter().all(|x| x.is_finite()));
                        fs::write(
                            out.join(format!("{name}-{prefix}.bin")),
                            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
                        )
                        .unwrap();
                    }
                }
                writeln!(log,"{}",json!({"mode":mode,"batch":batch_spec["name"],"draw":draw,"input_sha256":input_hash,"losses":losses,"seconds":started.elapsed().as_secs_f64()})).unwrap();
                log.flush().unwrap();
                count += 1;
                println!("centroid frozen {count}/20 {mode} {prefix}");
            }
        }
        for (name, _) in &parameters {
            let mut v = vec![0.; session.param_size(name).unwrap()];
            session.read_param(name, &mut v);
            assert_eq!(
                v,
                anchor.tensor_f32(name).unwrap(),
                "weight changed: {name}"
            );
        }
    }
    assert_eq!(count, 20);
    assert_eq!(crate::vision::checkpoint_sha256(&checkpoint).unwrap(), hash);
    fs::write(out.join("result.json"),serde_json::to_vec_pretty(&json!({"status":"complete","backward_passes":count,"optimizer_steps":0,"parameters_exact_after_each_component":70,"checkpoint_unchanged":true,"seconds":started.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
