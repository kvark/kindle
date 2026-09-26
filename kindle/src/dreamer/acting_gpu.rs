//! Categorical sampling and persistent live belief on the shared GPU queue.

use std::sync::Arc;

use blade_graphics::{self as gpu, ShaderBindable as _, ShaderData as _};
use meganeura::Session;

use super::{ActionMode, DreamerConfig};

struct SamplingBindings {
    logits: gpu::BufferPiece,
    draws: gpu::BufferPiece,
    onehot: gpu::BufferPiece,
    selected: gpu::BufferPiece,
    allowed: gpu::BufferPiece,
    params: [u32; 8],
}

impl gpu::ShaderData for SamplingBindings {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("logits", gpu::ShaderBinding::Buffer),
                ("draws", gpu::ShaderBinding::Buffer),
                ("onehot", gpu::ShaderBinding::Buffer),
                ("selected", gpu::ShaderBinding::Buffer),
                ("allowed", gpu::ShaderBinding::Buffer),
                ("params", gpu::ShaderBinding::Plain { size: 32 }),
            ],
        }
    }
    fn fill(&self, mut cx: gpu::PipelineContext) {
        self.logits.bind_to(&mut cx, 0);
        self.draws.bind_to(&mut cx, 1);
        self.onehot.bind_to(&mut cx, 2);
        self.selected.bind_to(&mut cx, 3);
        self.allowed.bind_to(&mut cx, 4);
        self.params.bind_to(&mut cx, 5);
    }
}

struct StateBindings {
    next_deter: gpu::BufferPiece,
    next_stoch: gpu::BufferPiece,
    active: gpu::BufferPiece,
    previous_deter: gpu::BufferPiece,
    previous_stoch: gpu::BufferPiece,
    feature: gpu::BufferPiece,
    params: [u32; 4],
}

impl gpu::ShaderData for StateBindings {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("next_deter", gpu::ShaderBinding::Buffer),
                ("next_stoch", gpu::ShaderBinding::Buffer),
                ("arrivals", gpu::ShaderBinding::Buffer),
                ("previous_deter", gpu::ShaderBinding::Buffer),
                ("previous_stoch", gpu::ShaderBinding::Buffer),
                ("feature", gpu::ShaderBinding::Buffer),
                ("params", gpu::ShaderBinding::Plain { size: 16 }),
            ],
        }
    }
    fn fill(&self, mut cx: gpu::PipelineContext) {
        self.next_deter.bind_to(&mut cx, 0);
        self.next_stoch.bind_to(&mut cx, 1);
        self.active.bind_to(&mut cx, 2);
        self.previous_deter.bind_to(&mut cx, 3);
        self.previous_stoch.bind_to(&mut cx, 4);
        self.feature.bind_to(&mut cx, 5);
        self.params.bind_to(&mut cx, 6);
    }
}

pub(super) struct ActingGpu {
    gpu: Arc<gpu::Context>,
    encoder: gpu::CommandEncoder,
    completion: Option<gpu::SyncPoint>,
    sample: gpu::ComputePipeline,
    commit: gpu::ComputePipeline,
    draws: gpu::Buffer,
    active: gpu::Buffer,
    onehot: gpu::Buffer,
    selected: gpu::Buffer,
    allowed: gpu::Buffer,
    streams: usize,
    deter: usize,
    stoch: usize,
    classes: usize,
    actions: usize,
    unimix: f32,
    actor_unimix: f32,
}

impl ActingGpu {
    pub fn new(gpu: Arc<gpu::Context>, config: &DreamerConfig, streams: usize) -> Self {
        let shape = config.network();
        let max_rows = streams * shape.stoch;
        let max_elements = streams * (shape.stoch * shape.classes).max(config.action_count);
        let sample_shader = gpu.create_shader(gpu::ShaderDesc {
            source: include_str!("categorical.wgsl"),
            naga_module: None,
        });
        let state_shader = gpu.create_shader(gpu::ShaderDesc {
            source: include_str!("live_state.wgsl"),
            naga_module: None,
        });
        let sample = gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "kindle_categorical",
            data_layouts: &[&SamplingBindings::layout()],
            compute: sample_shader.at("main"),
        });
        let commit = gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "kindle_live_state",
            data_layouts: &[&StateBindings::layout()],
            compute: state_shader.at("main"),
        });
        let buffer = |name, elements, memory| {
            gpu.create_buffer(gpu::BufferDesc {
                name,
                size: (elements * 4) as u64,
                memory,
            })
        };
        Self {
            encoder: gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "kindle_acting",
                buffer_count: 1,
                manual_barriers: false,
            }),
            completion: None,
            sample,
            commit,
            draws: buffer("kindle_live_uniforms", max_rows, gpu::Memory::Shared),
            active: buffer("kindle_live_active", streams, gpu::Memory::Shared),
            onehot: buffer("kindle_live_samples", max_elements, gpu::Memory::Device),
            selected: buffer("kindle_actions", max_rows, gpu::Memory::Device),
            allowed: buffer(
                "kindle_action_mask",
                streams * config.action_count,
                gpu::Memory::Shared,
            ),
            gpu,
            streams,
            deter: shape.deter,
            stoch: shape.stoch,
            classes: shape.classes,
            actions: config.action_count,
            unimix: config.unimix,
            actor_unimix: config.actor_unimix,
        }
    }

    pub fn posterior(
        &mut self,
        observe: &Session,
        policy: &Session,
        active: &[u32],
        draws: &[f32],
    ) {
        assert_eq!(active.len(), self.streams);
        assert_eq!(draws.len(), self.streams * self.stoch);
        self.wait();
        // These owned shared inputs are idle. All other live tensors stay device-side.
        unsafe {
            std::ptr::copy_nonoverlapping(active.as_ptr(), self.active.data().cast(), active.len());
        }
        self.upload_draws(draws);
        self.encoder.start();
        self.encode_sample(
            observe.output_buffer(1).unwrap(),
            draws.len(),
            self.classes,
            self.unimix,
            ActionMode::Sample,
            false,
        );
        let data = StateBindings {
            next_deter: observe.output_buffer(0).unwrap(),
            next_stoch: self.onehot.into(),
            active: self.active.into(),
            previous_deter: observe.input_buffer("previous_deter").unwrap(),
            previous_stoch: observe.input_buffer("previous_stoch").unwrap(),
            feature: policy.input_buffer("feature").unwrap(),
            params: [
                self.streams as u32,
                self.deter as u32,
                (self.stoch * self.classes) as u32,
                0,
            ],
        };
        {
            let mut pass = self.encoder.compute("kindle_commit_live_state");
            let mut pipeline = pass.with(&self.commit);
            pipeline.bind(0, &data);
            pipeline.dispatch([
                (self.streams * (self.deter + self.stoch * self.classes)).div_ceil(64) as u32,
                1,
                1,
            ]);
        }
        self.completion = Some(self.gpu.submit(&mut self.encoder));
    }

    pub fn actions(
        &mut self,
        policy: &Session,
        draws: &[f32],
        mode: ActionMode,
        mask: Option<&[bool]>,
    ) -> gpu::BufferPiece {
        assert_eq!(draws.len(), self.streams);
        self.wait();
        self.upload_draws(draws);
        if let Some(mask) = mask {
            assert_eq!(mask.len(), self.streams * self.actions);
            assert!(mask.chunks(self.actions).all(|row| row.iter().any(|&v| v)));
            let target = unsafe {
                std::slice::from_raw_parts_mut(self.allowed.data().cast::<u32>(), mask.len())
            };
            for (output, &enabled) in target.iter_mut().zip(mask) {
                *output = u32::from(enabled);
            }
        }
        self.encoder.start();
        self.encode_sample(
            policy.output_buffer(0).unwrap(),
            self.streams,
            self.actions,
            self.actor_unimix,
            mode,
            mask.is_some(),
        );
        self.completion = Some(self.gpu.submit(&mut self.encoder));
        self.selected.into()
    }

    fn encode_sample(
        &mut self,
        logits: gpu::BufferPiece,
        rows: usize,
        classes: usize,
        unimix: f32,
        mode: ActionMode,
        masked: bool,
    ) {
        let data = SamplingBindings {
            logits,
            draws: self.draws.into(),
            onehot: self.onehot.into(),
            selected: self.selected.into(),
            allowed: self.allowed.into(),
            params: [
                rows as u32,
                classes as u32,
                unimix.to_bits(),
                u32::from(mode == ActionMode::Greedy),
                u32::from(masked),
                0,
                0,
                0,
            ],
        };
        let mut pass = self.encoder.compute("kindle_categorical");
        let mut pipeline = pass.with(&self.sample);
        pipeline.bind(0, &data);
        pipeline.dispatch([rows.div_ceil(64) as u32, 1, 1]);
    }

    fn upload_draws(&self, draws: &[f32]) {
        assert!(draws.iter().all(|&draw| (0.0..1.0).contains(&draw)));
        unsafe {
            std::ptr::copy_nonoverlapping(draws.as_ptr(), self.draws.data().cast(), draws.len());
        }
    }

    pub fn wait(&mut self) {
        if let Some(done) = self.completion.take() {
            assert!(
                self.gpu
                    .wait_for(&done, !0)
                    .expect("acting GPU wait failed")
            );
        }
    }
}

impl Drop for ActingGpu {
    fn drop(&mut self) {
        self.wait();
        for buffer in [
            self.draws,
            self.active,
            self.onehot,
            self.selected,
            self.allowed,
        ] {
            self.gpu.destroy_buffer(buffer);
        }
        self.gpu.destroy_compute_pipeline(&mut self.sample);
        self.gpu.destroy_compute_pipeline(&mut self.commit);
        self.gpu.destroy_command_encoder(&mut self.encoder);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn acting_and_pixel_shaders_validate_without_a_gpu() {
        for source in [
            include_str!("categorical.wgsl"),
            include_str!("live_state.wgsl"),
            include_str!("../vision/preprocess.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
