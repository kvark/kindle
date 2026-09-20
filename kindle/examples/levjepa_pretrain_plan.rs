//! CPU-only allocation preflight. No GPU context, uploads or training.
//!
//! Usage: levjepa_pretrain_plan BATCH LOCAL_VIEWS NEW_REPORT.json
//! Estimates the disabled-cooperative-matrix path, including Adam moments.
//! Driver allocations, upload staging and EMA are not included.

use std::{error::Error, fs::OpenOptions, io::Write};

use kindle::vision::levjepa::pretrain::{Config, graph};
use meganeura::{CoopPolicy, SessionOptions};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.len() != 3 {
        return Err("usage: levjepa_pretrain_plan BATCH LOCAL_VIEWS NEW_REPORT.json".into());
    }
    let config = Config {
        batch: arguments[0].parse()?,
        local_views: arguments[1].parse()?,
        ..Default::default()
    };
    let graph = graph(config)?;
    let start = std::time::Instant::now();
    let (mut plan, _) = meganeura::compile_training_graph(&graph);
    let options = SessionOptions {
        coop: CoopPolicy::Disabled,
        ..Default::default()
    };
    let groups = meganeura::runtime::plan_dispatches(&mut plan, None, &options);
    let aliases = meganeura::memplan::plan_buffer_aliasing(&plan, &groups, None);
    let allocated: usize = aliases.sizes.iter().map(|size| (*size).max(4)).sum();
    let adam: usize = plan
        .param_grad_pairs
        .iter()
        .map(|(p, _)| plan.buffers[p.0 as usize].max(4) * 2)
        .sum();
    let mut kernel_counts = std::collections::BTreeMap::new();
    for dispatch in &plan.dispatches {
        *kernel_counts
            .entry(format!("{:?}", dispatch.shader))
            .or_insert(0) += 1;
    }
    let report = serde_json::json!({
        "kind": "cpu_plan_only", "coop": "Disabled", "config": config,
        "compile_seconds": start.elapsed().as_secs_f64(),
        "logical_bytes": plan.buffers.iter().sum::<usize>(),
        "physical_buffer_bytes": allocated, "adam_bytes": adam,
        "planned_bytes_with_adam": allocated + adam + 4,
        "excludes": ["driver allocations", "upload staging", "EMA"],
        "largest_buffer_bytes": plan.buffers.iter().max(),
        "physical_buffers": aliases.sizes.len(), "logical_buffers": plan.buffers.len(),
        "dispatches": plan.dispatches.len(), "barrier_groups": groups.len(),
        "parameter_gradient_pairs": plan.param_grad_pairs.len(),
        "kernel_counts": kernel_counts,
    });
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&arguments[2])?;
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    println!("{report}");
    Ok(())
}
