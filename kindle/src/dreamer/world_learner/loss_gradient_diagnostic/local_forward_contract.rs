//! Frozen local-history training/unrolled/streaming comparison at matched samples.

use super::*;

fn uses(graph: &Graph, name: &str) -> Vec<NodeId> {
    let weight = graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Parameter { name: p } if p == name))
        .unwrap()
        .id;
    graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::MatMul) && n.inputs[1] == weight)
        .map(|n| n.inputs[0])
        .collect()
}

fn traced_graph(c: &DreamerConfig, horizon: usize) -> (Graph, BTreeMap<String, usize>) {
    let mut graph =
        world::build_world_only_pixel_training_graph(c, horizon, PixelLoss::MotionEventRollout);
    let h = uses(&graph, "world.dynamics.prior0.weight");
    let z = uses(&graph, "world.decoder.sp1.weight");
    let logits = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Softplus { .. }))
        .map(|n| n.inputs[0])
        .filter(|&n| {
            graph.node(n).ty.shape == [c.batch_size * c.batch_length, c.observation_dim()]
                || graph.node(n).ty.shape == [c.batch_size, c.observation_dim()]
        })
        .collect::<Vec<_>>();
    assert_eq!(h.len(), c.batch_length + horizon);
    assert_eq!(z.len(), horizon + 1);
    assert_eq!(logits.len(), horizon + 1);
    assert_eq!(
        graph.node(z[0]).ty.shape,
        [
            c.batch_size * c.batch_length,
            c.network().stoch * c.network().classes
        ]
    );
    assert_eq!(
        graph.node(logits[0]).ty.shape,
        [c.batch_size * c.batch_length, c.observation_dim()]
    );
    let mut outputs = graph.outputs().to_vec();
    let mut layout = BTreeMap::new();
    for i in [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        world::LOSS_WORLD_ROLLOUT,
        world::LOSS_IMAGE_FEEDBACK,
        world::LOSS_IMAGE_FEEDBACK + 1,
    ] {
        layout.insert(format!("loss_{i}"), i);
    }
    for (prefix, nodes) in [("h", h), ("z", z), ("logits", logits)] {
        for (i, node) in nodes.into_iter().enumerate() {
            layout.insert(format!("{prefix}_{i}"), outputs.len());
            outputs.push(node);
        }
    }
    let logits = (1..=horizon)
        .map(|i| outputs[layout[&format!("logits_{i}")]])
        .collect::<Vec<_>>();
    for (i, logits) in logits.into_iter().enumerate() {
        let pixels = generated_pixels(&graph, logits, c.batch_size, c.observation_dim());
        layout.insert(format!("rgb_{}", i + 1), outputs.len());
        outputs.push(pixels);
    }
    graph.set_outputs(outputs);
    (graph, layout)
}

fn local_config(late: bool) -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = super::super::super::ObservationKind::Rgb32;
    c.batch_size = 2;
    c.batch_length = 4;
    c.world_backprop_length = 4;
    c.disagreement_bonus = false;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.rgb_spatial_decoder = true;
    c.rgb_spatial_fusion = true;
    c.rgb_spatial_action = true;
    c.rgb_late_action = late;
    c.world_action_features = Some(vec![vec![0.; 18]; 18]);
    c
}

fn generated_pixels(g: &Graph, logits: NodeId, batch: usize, width: usize) -> NodeId {
    let unique = |input, predicate: fn(&Op) -> bool, shape: &[usize]| {
        let found = g
            .nodes()
            .iter()
            .filter(|n| n.inputs.first() == Some(&input) && predicate(&n.op) && n.ty.shape == shape)
            .map(|n| n.id)
            .collect::<Vec<_>>();
        assert_eq!(found.len(), 1);
        found[0]
    };
    let sigmoid = unique(logits, |op| matches!(op, Op::Sigmoid), &[batch, width]);
    let flat = unique(
        sigmoid,
        |op| matches!(op, Op::Identity),
        &[batch * width, 1],
    );
    let centered = unique(flat, |op| matches!(op, Op::BiasAdd), &[batch * width, 1]);
    unique(centered, |op| matches!(op, Op::Identity), &[batch, width])
}

#[test]
fn local_trace_exposes_existing_nodes_without_changing_graph_or_original_outputs() {
    for late in [false, true] {
        let c = local_config(late);
        let original =
            world::build_world_only_pixel_training_graph(&c, 2, PixelLoss::MotionEventRollout);
        let (g, layout) = traced_graph(&c, 2);
        assert_eq!(g.nodes().len(), original.nodes().len());
        assert_eq!(&g.outputs()[..original.outputs().len()], original.outputs());
        let parameters = |g: &Graph| {
            g.nodes()
                .iter()
                .filter_map(|n| {
                    if let Op::Parameter { name } = &n.op {
                        Some((name.clone(), n.ty.shape.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(parameters(&g), parameters(&original));
        assert_eq!(parameters(&g).len(), 80);
        assert_eq!(
            g.node(g.outputs()[layout["h_0"]]).ty.shape,
            [2, c.network().deter]
        );
        assert_eq!(
            g.node(g.outputs()[layout["logits_0"]]).ty.shape,
            [8, c.observation_dim()]
        );
        for i in 1..=2 {
            assert_eq!(
                g.node(g.outputs()[layout[&format!("rgb_{i}")]]).ty.shape,
                [2, c.observation_dim()]
            );
        }
        for index in [
            world::LOSS_TOTAL,
            world::LOSS_WORLD_ROLLOUT,
            world::LOSS_IMAGE_FEEDBACK,
            world::LOSS_IMAGE_FEEDBACK + 1,
        ] {
            assert_eq!(layout[&format!("loss_{index}")], index);
        }
    }
}

#[test]
fn local_contract_feeds_use_physical_features_and_independent_reset_masks() {
    let mut c = local_config(true);
    let table = c.world_action_features.as_mut().unwrap();
    table[4][0] = 1.;
    table[3][1] = 1.;
    table[1][2] = 1.;
    let batch = Batch {
        observations: vec![0.; 8 * c.observation_dim()],
        actions: vec![-1, -1, 0, 4, 3, 1, 4, 0],
        rewards: vec![0.; 8],
        terminals: vec![false; 8],
    };
    let feeds = inputs(&c, &batch, 2, [95001, 96001]);
    let (graph, _) = traced_graph(&c, 2);
    validate_feeds(&graph, &feeds);
    assert!(
        feeds["previous_action_0"].iter().all(|&v| v == 0.)
            && feeds["keep_action_0"].iter().all(|&v| v == 0.)
    );
    assert!(feeds["previous_action_1"][..18].iter().all(|&v| v == 0.));
    assert_eq!(feeds["previous_action_1"][18], 1.);
    assert!(feeds["keep_action_1"].iter().all(|&v| v == 1.));
}

fn read(session: &Session, index: usize, length: usize) -> Vec<f32> {
    let mut values = vec![0.; length];
    session.read_output_by_index(index, &mut values);
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

fn frozen(session: &Session, anchor: &SafeTensorsModel) {
    for name in session.param_names() {
        let mut actual = vec![0.; session.param_size(name).unwrap()];
        session.read_param(name, &mut actual);
        let expected = anchor.tensor_f32(name).unwrap();
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}"
        );
    }
}

fn install(session: &mut Session, anchor: &SafeTensorsModel) {
    let names = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for name in names {
        session.set_parameter(&name, &anchor.tensor_f32(&name).unwrap());
    }
}

fn save(path: &Path, arrays: &BTreeMap<String, Vec<f32>>) {
    let mut file = BufWriter::new(File::create_new(path.with_extension("bin")).unwrap());
    let mut layout = Vec::new();
    let mut offset = 0;
    for (name, values) in arrays {
        assert!(values.iter().all(|v| v.is_finite()));
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        layout.push(json!({"name":name,"offset_f32":offset,"length":values.len()}));
        offset += values.len();
    }
    file.flush().unwrap();
    fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&layout).unwrap(),
    )
    .unwrap();
}

fn gumbel(logits: &[f32], uniforms: &[f32], classes: usize, mixing: f32) -> Vec<f32> {
    assert_eq!(logits.len(), uniforms.len());
    let mut result = vec![0.; logits.len()];
    let mut probabilities = vec![0.; classes];
    for ((l, u), out) in logits
        .chunks_exact(classes)
        .zip(uniforms.chunks_exact(classes))
        .zip(result.chunks_exact_mut(classes))
    {
        super::super::super::distributions::softmax_unimix(l, mixing, &mut probabilities);
        let mut best = f32::NEG_INFINITY;
        let mut selected = 0;
        for (i, (&p, &u)) in probabilities.iter().zip(u).enumerate() {
            let score = p.ln() - (-u.clamp(f32::MIN_POSITIVE, 1. - f32::EPSILON).ln()).ln();
            if score > best {
                best = score;
                selected = i;
            }
        }
        out[selected] = 1.;
    }
    result
}

fn decode(
    session: &mut Session,
    h: &[f32],
    z: &[f32],
    history: &[f32],
    action: &[f32],
    width: usize,
) -> Vec<f32> {
    session.set_input("decoder_history", history);
    session.set_input("decoder_action", action);
    session.set_input("deter", h);
    session.set_input("stoch", z);
    session.step();
    session.wait();
    read(session, 0, width)
}

fn stream(
    c: &DreamerConfig,
    feeds: &Inputs,
    train: &BTreeMap<String, Vec<f32>>,
    observe: &mut Session,
    transition: &mut Session,
    decoder: &mut Session,
) -> BTreeMap<String, Vec<f32>> {
    let b = c.batch_size;
    let size = c.network();
    let hw = b * size.deter;
    let zw = b * size.stoch * size.classes;
    let pixels = b * c.observation_dim();
    let mut h = vec![0.; hw];
    let mut z = vec![0.; zw];
    let mut prefix_h = Vec::new();
    let mut prefix_z = Vec::new();
    let mut rgb = Vec::new();
    let mut samples = Vec::new();
    let mut result = BTreeMap::new();
    let images = (0..32)
        .flat_map(|t| feeds[&format!("observation_{t}")].iter().copied())
        .collect::<Vec<_>>();
    let actions = (0..32)
        .flat_map(|t| {
            feeds[&format!("keep_action_{t}")]
                .chunks_exact(c.action_count)
                .map(|k| if k[0] == 0. { -1 } else { 0 })
        })
        .collect::<Vec<_>>();
    let mut history = super::super::super::spatial_history::after_cpu(
        &images,
        &actions,
        15,
        b,
        c.observation_dim(),
    );
    for t in 0..32 {
        observe.set_input("previous_deter", &h);
        observe.set_input("previous_stoch", &z);
        for name in [
            "previous_action",
            "keep_deter",
            "keep_stoch",
            "keep_action",
            "observation",
        ] {
            observe.set_input(name, &feeds[&format!("{name}_{t}")]);
        }
        observe.step();
        observe.wait();
        h = read(observe, 0, hw);
        let logits = read(observe, 1, zw);
        samples.extend(gumbel(
            &logits,
            &feeds[&format!("uniforms_{t}")],
            size.classes,
            c.unimix,
        ));
        z = train["z_0"][t * zw..(t + 1) * zw].to_vec();
        result.insert(format!("h_{t}"), h.clone());
        let observed_history = super::super::super::spatial_history::before_cpu(
            &images,
            &actions,
            t,
            b,
            c.observation_dim(),
        );
        rgb.extend(decode(
            decoder,
            &h,
            &z,
            &observed_history,
            &feeds[&format!("previous_action_{t}")],
            pixels,
        ));
        if t == 15 {
            prefix_h = h.clone();
            prefix_z = z.clone();
        }
    }
    result.insert("rgb_0".into(), rgb);
    result.insert("sample_0".into(), samples);
    h = prefix_h;
    z = prefix_z;
    for t in 0..16 {
        transition.set_input("deter", &h);
        transition.set_input("stoch", &z);
        transition.set_input("action", &feeds[&format!("previous_action_{}", t + 16)]);
        transition.step();
        transition.wait();
        h = read(transition, 0, hw);
        let logits = read(transition, 1, zw);
        result.insert(
            format!("sample_{}", t + 1),
            gumbel(
                &logits,
                &feeds[&format!("pixel_rollout_uniforms_{t}")],
                size.classes,
                c.unimix,
            ),
        );
        z = train[&format!("z_{}", t + 1)].clone();
        result.insert(format!("h_{}", t + 32), h.clone());
        let next = decode(
            decoder,
            &h,
            &z,
            &history,
            &feeds[&format!("previous_action_{}", t + 16)],
            pixels,
        );
        history =
            super::super::super::spatial_history::shift_cpu(&history, &next, c.observation_dim());
        result.insert(format!("rgb_{}", t + 1), next);
    }
    result
}

#[test]
#[ignore = "requires declared guarded frozen world forward-contract measurement"]
fn frozen_local_world_forward_contract() {
    let root = PathBuf::from(std::env::var("KINDLE_WORLD_LOCAL_FORWARD_ROOT").unwrap());
    let pins: Value =
        serde_json::from_slice(&fs::read(root.join("execution-pins.json")).unwrap()).unwrap();
    for (path, hash) in pins["pins"].as_object().unwrap() {
        assert_eq!(
            crate::vision::checkpoint_sha256(Path::new(path)).unwrap(),
            hash.as_str().unwrap()
        );
    }
    let settings: Value =
        serde_json::from_slice(&fs::read(root.join("settings.json")).unwrap()).unwrap();
    let out = root.join("native-results");
    fs::create_dir(&out).unwrap();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let start = Instant::now();
    let mut cases = 0;
    let mut dimensions = BTreeMap::new();
    for model in settings["models"].as_array().unwrap() {
        let c: DreamerConfig = serde_json::from_value(model["config"].clone()).unwrap();
        validate_config(&c).unwrap();
        c.validate();
        assert_eq!(
            (c.batch_size, c.batch_length, c.world_backprop_length),
            (8, 32, 32)
        );
        assert!(
            !c.world_mean_latents
                && !c.world_gaussian
                && c.rgb_spatial_decoder
                && c.rgb_spatial_fusion
                && c.rgb_spatial_action
                && c.world_action_features.is_some()
                && c.color_centroid_supervision.is_none()
                && c.rgb_color_attention.is_none()
        );
        let checkpoint = PathBuf::from(model["checkpoint"].as_str().unwrap());
        let hash = crate::vision::checkpoint_sha256(&checkpoint).unwrap();
        let anchor = SafeTensorsModel::load(checkpoint.clone()).unwrap();
        assert_eq!(anchor.tensor_info().len(), 240);
        let (graph, layout) = traced_graph(&c, 16);
        let mut train = runtime::build_training_session(&graph, &gpu, &c);
        train.clear_optimizer();
        install(&mut train, &anchor);
        let mut unrolled = runtime::build_learner_inference_session(&graph, &gpu, &c);
        install(&mut unrolled, &anchor);
        let mut observe =
            runtime::build_learner_inference_session(&world::build_observe_graph(&c, 8), &gpu, &c);
        install(&mut observe, &anchor);
        let mut transition = runtime::build_learner_inference_session(
            &world::build_transition_graph(&c, 8),
            &gpu,
            &c,
        );
        install(&mut transition, &anchor);
        let mut decoder = runtime::build_learner_inference_session(
            &world::build_observation_prediction_graph(&c, 8),
            &gpu,
            &c,
        );
        install(&mut decoder, &anchor);
        assert_eq!(train.param_names().len(), 80);
        dimensions.insert(model["name"].as_str().unwrap().to_owned(),json!({"deter":c.network().deter,"stoch":c.network().stoch,"classes":c.network().classes}));
        for batch in settings["batches"].as_array().unwrap() {
            let data = load_batch(batch["path"].as_str().unwrap());
            for (draw, seeds) in settings["seeds"].as_array().unwrap().iter().enumerate() {
                let feeds = inputs(
                    &c,
                    &data,
                    16,
                    [seeds[0].as_u64().unwrap(), seeds[1].as_u64().unwrap()],
                );
                validate_feeds(&graph, &feeds);
                let prefix = format!(
                    "{}-{}-{draw}",
                    model["name"].as_str().unwrap(),
                    batch["name"].as_str().unwrap()
                );
                let mut actual = BTreeMap::new();
                for (name, session) in [("training", &mut train), ("unrolled", &mut unrolled)] {
                    for (key, value) in &feeds {
                        session.set_input(key, value);
                    }
                    session.step();
                    session.wait();
                    let arrays = layout
                        .iter()
                        .map(|(key, &index)| {
                            (
                                key.clone(),
                                read(
                                    session,
                                    index,
                                    graph.node(graph.outputs()[index]).ty.num_elements(),
                                ),
                            )
                        })
                        .collect::<BTreeMap<_, _>>();
                    save(&out.join(format!("{prefix}-{name}")), &arrays);
                    frozen(session, &anchor);
                    if name == "training" {
                        actual = arrays;
                    }
                }
                let streamed = stream(
                    &c,
                    &feeds,
                    &actual,
                    &mut observe,
                    &mut transition,
                    &mut decoder,
                );
                save(&out.join(format!("{prefix}-streaming")), &streamed);
                for session in [&train, &unrolled, &observe, &transition, &decoder] {
                    frozen(session, &anchor);
                }
                assert_eq!(train.adam_step_count(), 0);
                assert_eq!(unrolled.adam_step_count(), 0);
                cases += 1;
                eprintln!("Completed frozen forward case {prefix}");
            }
        }
        assert_eq!(crate::vision::checkpoint_sha256(&checkpoint).unwrap(), hash);
    }
    let result = json!({"status":"complete","cases":cases,"backward_passes":cases,"optimizer_steps":0,"parameters_exact_after_each_case":80,"dimensions":dimensions,"checkpoints_unchanged":true,"seconds":start.elapsed().as_secs_f64()});
    assert_eq!(cases, 8);
    fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("{result}");
}
