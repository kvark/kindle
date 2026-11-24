//! Paged device replay. Collection copies features/state on GPU; only sampled
//! training batches cross the existing learner's host target-building boundary.

use std::sync::Arc;

use blade_graphics::{self as gpu, ShaderBindable as _, ShaderData as _};
use meganeura::Session;

use super::{DreamerConfig, device_copy::DeviceCopies, readback::Readback};

const PAGE_FRAMES: usize = 256;

struct PackBindings {
    values: gpu::BufferPiece,
    packed: gpu::BufferPiece,
    params: [u32; 16],
}

impl gpu::ShaderData for PackBindings {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("values", gpu::ShaderBinding::Buffer),
                ("packed", gpu::ShaderBinding::Buffer),
                ("params", gpu::ShaderBinding::Plain { size: 64 }),
            ],
        }
    }

    fn fill(&self, mut context: gpu::PipelineContext) {
        self.values.bind_to(&mut context, 0);
        self.packed.bind_to(&mut context, 1);
        self.params.bind_to(&mut context, 2);
    }
}

/// Packs centered RGB values into bytes on the shared queue.
struct Packer {
    encoder: gpu::CommandEncoder,
    pipeline: gpu::ComputePipeline,
    completion: Option<gpu::SyncPoint>,
}

impl Packer {
    fn new(gpu: &gpu::Context) -> Self {
        let shader = gpu.create_shader(gpu::ShaderDesc {
            source: include_str!("replay_pack.wgsl"),
            naga_module: None,
        });
        let pipeline = gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            full_subgroup_size: None,
            name: "kindle_replay_pack",
            data_layouts: &[&PackBindings::layout()],
            compute: shader.at("main"),
        });
        Self {
            encoder: gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "kindle_replay_pack",
                buffer_count: 1,
                manual_barriers: false,
            }),
            pipeline,
            completion: None,
        }
    }

    fn wait(&mut self, gpu: &gpu::Context) {
        if let Some(done) = self.completion.take() {
            assert!(gpu.wait_for(&done, !0).expect("replay pack completion"));
        }
    }
}

pub(super) struct DeviceReplay {
    gpu: Arc<gpu::Context>,
    copies: DeviceCopies,
    pub readback: Readback,
    pages: Vec<gpu::Buffer>,
    capacity: usize,
    width: usize,
    /// Words one stored observation occupies in a replay row.
    observation: usize,
    /// Centered values in one observation.
    observation_values: usize,
    packer: Option<Packer>,
    deter: usize,
    state: usize,
    pixels: usize,
}

impl DeviceReplay {
    pub fn new(gpu: Arc<gpu::Context>, config: &DreamerConfig) -> Self {
        Self {
            copies: DeviceCopies::new(Arc::clone(&gpu)),
            readback: Readback::new(Arc::clone(&gpu)),
            packer: config.replay_compact_rgb.then(|| Packer::new(&gpu)),
            gpu,
            pages: Vec::new(),
            capacity: config.replay_capacity,
            width: config.replay_observation_words()
                + config.feature_dim()
                + if config.video_encoder.is_some() {
                    crate::vision::levjepa::joint::PIXELS
                } else {
                    0
                },
            observation: config.replay_observation_words(),
            observation_values: config.observation_dim(),
            deter: config.network().deter,
            state: config.feature_dim(),
            pixels: if config.video_encoder.is_some() {
                crate::vision::levjepa::joint::PIXELS
            } else {
                0
            },
        }
    }

    fn allocate_through(&mut self, last: usize) {
        assert!(last < self.capacity);
        while self.pages.len() <= last / PAGE_FRAMES {
            let rows = PAGE_FRAMES.min(self.capacity - self.pages.len() * PAGE_FRAMES);
            self.pages.push(self.gpu.create_buffer(gpu::BufferDesc {
                name: "kindle_replay",
                size: (rows * self.width * 4) as u64,
                memory: gpu::Memory::Device,
            }));
        }
    }

    pub fn store_pixels(&mut self, session: &Session, slots: &[(usize, usize)]) {
        if slots.is_empty() {
            return;
        }
        assert!(self.pixels > 0);
        self.allocate_through(slots.iter().map(|&(_, slot)| slot).max().unwrap());
        let copies = slots
            .iter()
            .map(|&(stream, slot)| {
                let mut source = session.input_buffer("patches").unwrap();
                source.offset += (stream * self.pixels * 4) as u64;
                (
                    source,
                    self.region(slot, self.observation + self.state),
                    self.pixels * 4,
                )
            })
            .collect::<Vec<_>>();
        self.copies.copy_regions(&copies);
    }

    pub fn store(&mut self, session: &Session, slots: &[(usize, usize)]) {
        if slots.is_empty() {
            return;
        }
        self.allocate_through(slots.iter().map(|&(_, slot)| slot).max().unwrap());
        let mut copies = Vec::with_capacity(slots.len() * 3);
        if let Some(packer) = &mut self.packer {
            packer.wait(&self.gpu);
            let values = session.input_buffer("observation").unwrap();
            assert!(
                values.offset
                    + (slots.iter().map(|&(stream, _)| stream + 1).max().unwrap()
                        * self.observation_values
                        * 4) as u64
                    <= values.buffer.size()
            );
            packer.encoder.start();
            {
                let mut pass = packer.encoder.compute("kindle_replay_pack");
                let mut pipeline = pass.with(&packer.pipeline);
                for &(stream, slot) in slots {
                    let mut params = [0; 16];
                    params[0] = (stream * self.observation_values) as u32;
                    params[1] = (slot % PAGE_FRAMES * self.width) as u32;
                    params[2] = self.observation as u32;
                    pipeline.bind(
                        0,
                        &PackBindings {
                            values,
                            packed: self.pages[slot / PAGE_FRAMES].into(),
                            params,
                        },
                    );
                    pipeline.dispatch([(self.observation as u32).div_ceil(64), 1, 1]);
                }
            }
            packer.completion = Some(self.gpu.submit(&mut packer.encoder));
        }
        for &(stream, slot) in slots {
            for (name, offset, width) in [
                ("observation", 0, self.observation),
                ("previous_deter", self.observation, self.deter),
                (
                    "previous_stoch",
                    self.observation + self.deter,
                    self.state - self.deter,
                ),
            ] {
                if name == "observation" && self.packer.is_some() {
                    continue;
                }
                let mut source = session.input_buffer(name).unwrap();
                source.offset += (stream * width * 4) as u64;
                copies.push((source, self.region(slot, offset), width * 4));
            }
        }
        self.copies.copy_regions(&copies);
    }

    pub fn region(&self, slot: usize, offset: usize) -> gpu::BufferPiece {
        assert!(slot < self.capacity && offset < self.width);
        self.pages[slot / PAGE_FRAMES].at(((slot % PAGE_FRAMES * self.width + offset) * 4) as u64)
    }

    pub fn wait(&mut self) {
        self.copies.wait();
        if let Some(packer) = &mut self.packer {
            packer.wait(&self.gpu);
        }
    }
}

impl Drop for DeviceReplay {
    fn drop(&mut self) {
        self.wait();
        if let Some(mut packer) = self.packer.take() {
            self.gpu.destroy_command_encoder(&mut packer.encoder);
            self.gpu.destroy_compute_pipeline(&mut packer.pipeline);
        }
        for page in self.pages.drain(..) {
            self.gpu.destroy_buffer(page);
        }
    }
}
