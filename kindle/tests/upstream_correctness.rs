//! Upstream numerical assertions are unchanged; observe the same NVIDIA device
//! before and after each explicitly selected test. These are not peak-memory
//! measurements. No GPU work occurs merely by listing or building these tests.

struct DeviceBudget {
    gpu: blade_graphics::Context,
    case: &'static str,
}

impl DeviceBudget {
    fn new(case: &'static str) -> Self {
        assert_eq!(
            std::env::var("KINDLE_GPU_DRIVER").as_deref(),
            Ok("580.178.04")
        );
        let options = meganeura::GpuOptions::from_env();
        assert_eq!(options.device_id, Some(0x2c02));
        assert!(!options.timing && !options.capture);
        for key in ["VK_DRIVER_FILES", "VK_ICD_FILENAMES"] {
            assert_eq!(
                std::env::var(key).as_deref(),
                Ok("/usr/share/vulkan/icd.d/nvidia_icd.json")
            );
        }
        let gpu = meganeura::init_gpu_context_with(options).expect("declared NVIDIA device");
        let this = Self { gpu, case };
        this.check("before");
        this
    }

    fn check(&self, phase: &str) {
        let device = self.gpu.device_information();
        assert_eq!(device.device_name, "NVIDIA GeForce RTX 5080");
        assert_eq!(device.driver_name, "NVIDIA");
        assert_eq!(device.driver_info, "580.178.04");
        assert!(!device.is_software_emulated);
        let memory = self.gpu.memory_stats();
        eprintln!(
            "upstream_device_budget={}",
            serde_json::json!({
                "case": self.case, "phase": phase,
                "device_name": device.device_name, "driver_name": device.driver_name,
                "driver_info": device.driver_info, "software": device.is_software_emulated,
                "budget_bytes": memory.budget, "usage_bytes": memory.usage,
            })
        );
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 * 1024 * 1024 * 1024);
    }
}

impl Drop for DeviceBudget {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.check("after");
        }
    }
}

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/oracle/basic.rs"]
mod basic;

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/oracle/norm.rs"]
mod norm;

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/oracle/regressions.rs"]
mod regressions;

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/oracle/autodiff.rs"]
mod autodiff;

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/oracle/blocks.rs"]
mod blocks;

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/oracle/losses.rs"]
mod losses;

#[path = "/x/Code/kindle/runs/meganeura-correctness-gpu-20260924.y1vWOG/observed-upstream/shader_audit.rs"]
mod shader_audit;
