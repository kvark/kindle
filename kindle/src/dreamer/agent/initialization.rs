//! Isolated, explicitly selected initialization diagnostic. Never trains or acts.

use super::*;
use crate::vision::levjepa::{self, LeVJepaPerception};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

const STREAMS: usize = 6;
const SELECTION: &str = "combined-driver-control-20260916";

fn config() -> DreamerConfig {
    let config: DreamerConfig =
        serde_json::from_str(include_str!("initialization_config.json")).unwrap();
    config.validate();
    config
}

fn write_mark(writer: &mut impl Write, phase: &str, data: serde_json::Value) -> io::Result<()> {
    serde_json::to_writer(
        &mut *writer,
        &serde_json::json!({
            "kindle_initialization": 1,
            "pid": std::process::id(),
            "unix_ns": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos(),
            "phase": phase,
            "data": data,
        }),
    )?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn mark(phase: &str, data: serde_json::Value) {
    write_mark(&mut io::stderr().lock(), phase, data)
        .expect("initialization fixture trace could not be flushed");
}

fn declared_driver(value: Option<&str>) -> &'static str {
    match value {
        Some("580.178.04") => "580.178.04",
        Some("595.91.07") => "595.91.07",
        _ => panic!("a declared driver version is required"),
    }
}

fn check_device(info: &crate::GpuDeviceInfo, expected_driver: &str) {
    assert!(!info.is_software_emulated, "software adapter refused");
    assert_eq!(info.device_name, "NVIDIA GeForce RTX 5080");
    assert_eq!(info.driver_name, "NVIDIA");
    assert_eq!(info.driver_info, expected_driver);
    assert_eq!(info.requested_device_id.as_deref(), Some("0x2c02"));
}

// Keep the same eleven CPU graphs live, in the production constructor's order,
// but create only the first GPU world session. No substitute tiny graph is used.
fn graphs(config: &DreamerConfig) -> Vec<meganeura::Graph> {
    let starts = config.batch_size * config.batch_length;
    let imagined_rows = starts * config.imagination_length;
    let replay_rows = config.batch_size * (config.batch_length - 1);

    let world_train_config = world_training_config(config);
    let world_train_graph =
        world::build_training_graph(&world_train_config, config.world_backprop_length);
    let world_observe_batch_graph = world::build_observe_graph(config, config.batch_size);
    let world_observe_live_graph = world::build_observe_graph(config, 1);
    let world_transition_graph = world::build_transition_graph(config, starts);
    let world_transition_live_graph = world::build_transition_graph(config, 1);
    let world_head_graph = world::build_imagination_head_graph(config, starts);
    let world_head_live_graph = world::build_head_graph(config, 1);
    let behavior_train_graph = behavior::build_training_graph(config, imagined_rows, replay_rows);
    let behavior_online_graph = behavior::build_inference_graph(config, starts);
    let behavior_slow_graph = behavior::build_value_inference_graph(config, starts);
    let policy_live_graph = behavior::build_actor_inference_graph(config, 1);
    vec![
        world_train_graph,
        world_observe_batch_graph,
        world_observe_live_graph,
        world_transition_graph,
        world_transition_live_graph,
        world_head_graph,
        world_head_live_graph,
        behavior_train_graph,
        behavior_online_graph,
        behavior_slow_graph,
        policy_live_graph,
    ]
}

#[test]
#[ignore = "requires a separately declared guarded initialization-only GPU invocation"]
fn combined_frontend_world_initialization_only() {
    assert_eq!(
        std::env::var("KINDLE_INIT_DIAGNOSTIC").as_deref(),
        Ok(SELECTION)
    );
    let driver = declared_driver(std::env::var("KINDLE_INIT_EXPECTED_DRIVER").ok().as_deref());
    let checkpoint = std::env::var_os("KINDLE_INIT_ENCODER").expect("pinned encoder path required");
    let checkpoint = Path::new(&checkpoint);
    let identity = PerceptionKind::LeVJepa.identity(levjepa::CHECKPOINT_SHA256.into());
    identity.verify_file(checkpoint).unwrap();
    let config = config();
    mark(
        "begin",
        serde_json::json!({"streams": STREAMS, "config": config, "perception": identity}),
    );

    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let device = crate::gpu_device_info(gpu.device_information());
    check_device(&device, driver);
    mark("device", serde_json::to_value(&device).unwrap());
    mark("frontend.before", serde_json::json!({}));
    let perception =
        LeVJepaPerception::load_batched(checkpoint, STREAMS, Some(Arc::clone(&gpu)), None).unwrap();
    check_device(&perception.gpu_device(), driver);
    mark("frontend.ready", serde_json::json!({}));

    mark("graphs.before", serde_json::json!({}));
    let graphs = graphs(&config);
    mark(
        "graphs.ready",
        serde_json::json!({
            "nodes": graphs.iter().map(|graph| graph.nodes().len()).collect::<Vec<_>>(),
        }),
    );
    mark("world.before", serde_json::json!({}));
    let world = build_session(&graphs[0], &gpu, Mode::Training, config.skip_full_optimize);
    check_device(&crate::gpu_device_info(world.device_information()), driver);
    mark("world.ready", serde_json::json!({}));

    // Explicit drops retain the frontend and all CPU graphs through world build.
    mark("drop.before", serde_json::json!({}));
    drop(world);
    drop(graphs);
    drop(perception);
    drop(gpu);
    mark(
        "complete",
        serde_json::json!({"actions": 0, "updates": 0, "gpu_sessions": 2}),
    );
}

#[test]
fn declared_pixel_configuration_is_not_a_tiny_fixture() {
    let config = config();
    assert_eq!(STREAMS, 6);
    assert_eq!(config.action_count, 18);
    assert_eq!(config.batch_size, 16);
    assert_eq!(config.batch_length, 64);
    assert_eq!(config.world_backprop_length, 64);
    assert_eq!(config.world_microbatch_size(), 16);
    assert_eq!(config.train_ratio, 256.0);
    assert_eq!(config.seed, 7301);
    assert!(!config.skip_full_optimize);
    assert_eq!(config.loss_scales.future_prediction, 0.25);
    assert_eq!(config.loss_scales.reconstruction, 0.0);
}

#[test]
fn builds_only_cpu_graphs_without_a_device() {
    let mut config = DreamerConfig::tiny(3);
    config.batch_size = 2;
    config.batch_length = 4;
    config.world_backprop_length = 4;
    let graphs = graphs(&config);
    assert_eq!(graphs.len(), 11);
    assert!(graphs.iter().all(|graph| !graph.nodes().is_empty()));
}

#[test]
fn fixture_records_are_single_line_and_flushed() {
    #[derive(Default)]
    struct Sink {
        bytes: Vec<u8>,
        flushes: usize,
    }
    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }
    let mut sink = Sink::default();
    write_mark(&mut sink, "a\nb", serde_json::json!({})).unwrap();
    assert_eq!(sink.flushes, 1);
    assert_eq!(sink.bytes.iter().filter(|&&byte| byte == b'\n').count(), 1);
    let row: serde_json::Value = serde_json::from_slice(&sink.bytes).unwrap();
    assert_eq!(row["phase"], "a\nb");
    assert_eq!(row["kindle_initialization"], 1);
}

#[test]
fn fixture_records_propagate_write_failure() {
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("must not flush after write failure");
        }
    }
    assert!(write_mark(&mut Broken, "begin", serde_json::json!({})).is_err());
}

#[test]
fn device_gate_rejects_software_and_other_drivers() {
    for driver in ["580.178.04", "595.91.07"] {
        let good = crate::GpuDeviceInfo {
            device_name: "NVIDIA GeForce RTX 5080".into(),
            driver_name: "NVIDIA".into(),
            driver_info: driver.into(),
            is_software_emulated: false,
            requested_device_id: Some("0x2c02".into()),
        };
        check_device(&good, driver);
        for field in 0..5 {
            let mut wrong = good.clone();
            match field {
                0 => wrong.device_name = "another device".into(),
                1 => wrong.driver_name = "another driver".into(),
                2 => {
                    wrong.driver_info = if driver == "580.178.04" {
                        "595.91.07".into()
                    } else {
                        "580.178.04".into()
                    };
                }
                3 => wrong.is_software_emulated = true,
                _ => wrong.requested_device_id = None,
            }
            assert!(std::panic::catch_unwind(|| check_device(&wrong, driver)).is_err());
        }
    }
}

#[test]
fn driver_declaration_rejects_missing_or_unlisted_versions() {
    for driver in ["580.178.04", "595.91.07"] {
        assert_eq!(declared_driver(Some(driver)), driver);
    }
    for value in [None, Some(""), Some("595.71.05"), Some("latest")] {
        assert!(std::panic::catch_unwind(|| declared_driver(value)).is_err());
    }
}
