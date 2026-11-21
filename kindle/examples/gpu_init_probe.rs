//! One context, three small allocations and one independently checked dispatch.
//! Run only under the host guard; no model, replay, training or NVML queries.

use blade_graphics::{BufferDesc, Memory};
use std::{io::Write, sync::Arc, time::SystemTime};

struct Logger;

fn timestamp() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_micros()
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{} {} {}", timestamp(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

fn stage(name: &str, gpu: Option<&blade_graphics::Context>) {
    let mut row = serde_json::json!({"stage": name, "unix_us": timestamp()});
    if let Some(gpu) = gpu {
        let stats = gpu.memory_stats();
        row["usage_bytes"] = stats.usage.into();
        row["budget_bytes"] = stats.budget.into();
        assert!(stats.budget.saturating_sub(stats.usage) >= 2 << 30);
    }
    println!("{row}");
    std::io::stdout().flush().unwrap();
}

fn main() {
    log::set_logger(&Logger).unwrap();
    log::set_max_level(log::LevelFilter::Info);
    let expected =
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").expect("explicit native device required");
    stage("before_context", None);
    let gpu = Arc::new(kindle::init_gpu_context().expect("native context initialization"));
    assert!(!gpu.device_information().is_software_emulated);
    assert_eq!(gpu.device_information().device_name, expected);
    stage("context_ready", Some(&gpu));
    for memory in [Memory::Device, Memory::Shared, Memory::Upload] {
        stage(
            &format!("before_4096_byte_{memory:?}_allocation"),
            Some(&gpu),
        );
        let buffer = gpu.create_buffer(BufferDesc {
            name: "initialization probe",
            size: 4096,
            memory,
        });
        stage(
            &format!("after_4096_byte_{memory:?}_allocation"),
            Some(&gpu),
        );
        gpu.destroy_buffer(buffer);
    }
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[256]);
    let y = graph.mul(x, x);
    graph.set_outputs(vec![y]);
    stage("before_session", Some(&gpu));
    let config = meganeura::SessionConfig::inference_from_env_on(Arc::clone(&gpu));
    let (mut session, _) = meganeura::build(&graph, config);
    stage("session_ready", Some(&gpu));
    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
    session.set_input("x", &input);
    stage("before_dispatch", Some(&gpu));
    session.step();
    session.wait();
    stage("dispatch_complete", Some(&gpu));
    let output = session.read_output(256);
    assert_eq!(output, input.iter().map(|x| x * x).collect::<Vec<_>>());
    stage("256_outputs_exact", Some(&gpu));
    drop(session);
    stage("session_dropped", Some(&gpu));
    drop(gpu);
    stage("context_dropped", None);
}
