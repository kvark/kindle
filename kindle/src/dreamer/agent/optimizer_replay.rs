//! Replay fixed raw gradients through the optimizer, outside any learner.

use super::*;
use meganeura::Graph;
use std::io::Write;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct Manifest {
    config: DreamerConfig,
    checkpoint: PathBuf,
    parameters: Vec<(String, Vec<usize>)>,
    gradients: Vec<PathBuf>,
    learner_step: u64,
}

fn optimizer_graph(parameters: &[(String, Vec<usize>)]) -> Graph {
    let mut graph = Graph::new();
    let mut losses = Vec::new();
    for (index, (name, shape)) in parameters.iter().enumerate() {
        let parameter = graph.parameter(name, shape);
        let gradient = graph.input(&format!("gradient_{index}"), shape);
        let product = graph.mul(parameter, gradient);
        losses.push(graph.sum_all(product));
    }
    let loss = super::super::networks::sum(&mut graph, &losses);
    graph.set_outputs(vec![loss]);
    graph
}

#[test]
fn optimizer_graph_preserves_scalar_loss_and_parameters() {
    let mut cases = vec![vec![
        ("weight".into(), vec![2, 3]),
        ("bias".into(), vec![3]),
        ("kernel".into(), vec![2, 2, 2, 2]),
    ]];
    if let Some(path) = std::env::var_os("KINDLE_OPTIMIZER_CPU_INPUTS") {
        let manifest: Manifest =
            serde_json::from_slice(&fs::read(PathBuf::from(path).join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.parameters.len(), 86);
        cases.push(manifest.parameters);
    }
    for parameters in cases {
        let graph = optimizer_graph(&parameters);
        assert_eq!(graph.node(graph.outputs()[0]).ty.shape, [1]);
        let (plan, _) = meganeura::train::compile_training_graph(&graph);
        assert_eq!(plan.param_grad_pairs.len(), parameters.len());
        assert_eq!(plan.param_buffers.len(), parameters.len());
        assert_eq!(plan.input_buffers.len(), parameters.len());
        for (name, buffer) in &plan.param_buffers {
            let (_, shape) = parameters.iter().find(|p| p.0 == *name).unwrap();
            assert_eq!(&plan.param_types[buffer].shape, shape);
        }
    }
}

fn read_floats(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

fn write_json(path: &Path, value: &impl serde::Serialize) {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), value).unwrap();
}

#[test]
#[ignore = "requires guarded RX7900XT and pinned saved gradients/optimizer state"]
fn recorded_gradients_match_optimizer_estimates() {
    let input = PathBuf::from(std::env::var_os("KINDLE_OPTIMIZER_INPUTS").unwrap());
    let output = PathBuf::from(std::env::var_os("KINDLE_OPTIMIZER_OUTPUT").unwrap());
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(input.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.learner_step, 12477);
    assert_eq!(manifest.parameters.len(), 86);
    assert_eq!(manifest.gradients.len(), 2);
    assert_eq!(
        manifest.config.training_precision,
        super::super::config::TrainingPrecision::Auto
    );
    assert!(manifest.parameters.windows(2).all(|p| p[0].0 < p[1].0));
    let elements: usize = manifest
        .parameters
        .iter()
        .map(|(_, shape)| shape.iter().product::<usize>())
        .sum();
    assert_eq!(elements, 14_688_256);
    let graph = optimizer_graph(&manifest.parameters);
    fs::create_dir(&output).unwrap();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let device = crate::gpu_device_info(gpu.device_information());
    assert_eq!(
        device.device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    assert!(!device.is_software_emulated);
    let mut session = super::super::runtime::build_training_session(&graph, &gpu, &manifest.config);
    assert_eq!(session.param_names().len(), manifest.parameters.len());
    let started = Instant::now();
    let mut minimum_headroom = u64::MAX;
    let mut batches = Vec::new();
    for (batch, gradient_path) in manifest.gradients.iter().enumerate() {
        let path = output.join(format!("batch-{batch}"));
        fs::create_dir(&path).unwrap();
        checkpoint::load_session(&mut session, &manifest.checkpoint).unwrap();
        session.clear_optimizer();
        session.clear_grad_accumulate();
        session.disable_grad_clip();
        session.zero_grad();
        let stats = gpu.memory_stats();
        let headroom = stats.budget.saturating_sub(stats.usage);
        assert!(headroom >= 2 << 30);
        minimum_headroom = minimum_headroom.min(headroom);
        session
            .save_checkpoint(&path.join("before.safetensors"))
            .unwrap();
        let gradients = read_floats(gradient_path);
        assert_eq!(gradients.len(), elements);
        let mut offset = 0;
        for (index, (name, shape)) in manifest.parameters.iter().enumerate() {
            let count: usize = shape.iter().product();
            assert!(session.has_param_grad(name));
            assert_eq!(session.param_size(name), Some(count));
            session.set_input(
                &format!("gradient_{index}"),
                &gradients[offset..offset + count],
            );
            offset += count;
        }
        session.step();
        session.wait();
        assert!(session.read_loss().is_finite());
        let mut offset = 0;
        for (name, shape) in &manifest.parameters {
            let count: usize = shape.iter().product();
            let mut actual = vec![0.0; count];
            session.read_param_grad(name, &mut actual);
            assert_eq!(
                actual,
                gradients[offset..offset + count],
                "raw derivative: {name}"
            );
            offset += count;
        }
        configure_d3_optimizer(
            &mut session,
            &manifest.config,
            manifest.learner_step,
            manifest.config.learning_rate,
        );
        session.step();
        session.wait();
        assert!(session.read_loss().is_finite());
        session
            .save_checkpoint(&path.join("after.safetensors"))
            .unwrap();
        let mut clipped = std::io::BufWriter::new(
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path.join("clipped.f32"))
                .unwrap(),
        );
        for (name, shape) in &manifest.parameters {
            let mut values = vec![0.0; shape.iter().product()];
            session.read_param_grad(name, &mut values);
            assert!(values.iter().all(|v| v.is_finite()));
            for value in values {
                clipped.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        clipped.flush().unwrap();
        batches.push(
            serde_json::json!({"index":batch,"raw_gradients_exact":true,"optimizer_steps":1}),
        );
        eprintln!("optimizer batch {batch} complete; original checkpoint retained");
    }
    write_json(
        &output.join("result.json"),
        &serde_json::json!({
        "status":"complete","seconds":started.elapsed().as_secs_f64(),"batches":batches,
        "parameters":86,"elements":elements,"optimizer_steps":2,"synthetic_backward_passes":4,
        "world_backward_passes":0,"learner_updates":0,"gameplay_actions":0,
        "minimum_sampled_headroom":minimum_headroom}),
    );
}
