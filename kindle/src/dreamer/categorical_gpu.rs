//! Detached categorical means on the same queue as the learner.
use std::sync::Arc;

use blade_graphics::{self as gpu, ShaderBindable as _, ShaderData as _};
use meganeura::{Session, runtime::ExternalSlot};

use super::distributions::TwoHotBins;

struct Bindings {
    logits: gpu::BufferPiece,
    bins: gpu::BufferPiece,
    means: gpu::BufferPiece,
    params: [u32; 4],
}

impl gpu::ShaderData for Bindings {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("logits", gpu::ShaderBinding::Buffer),
                ("bins", gpu::ShaderBinding::Buffer),
                ("means", gpu::ShaderBinding::Buffer),
                ("params", gpu::ShaderBinding::Plain { size: 16 }),
            ],
        }
    }

    fn fill(&self, mut context: gpu::PipelineContext) {
        self.logits.bind_to(&mut context, 0);
        self.bins.bind_to(&mut context, 1);
        self.means.bind_to(&mut context, 2);
        self.params.bind_to(&mut context, 3);
    }
}

pub(super) struct CategoricalGpu {
    gpu: Arc<gpu::Context>,
    encoder: gpu::CommandEncoder,
    pipeline: gpu::ComputePipeline,
    completion: Option<gpu::SyncPoint>,
    bins: gpu::Buffer,
    count: usize,
    output: gpu::Buffer,
    rows: usize,
}

impl CategoricalGpu {
    pub fn new(gpu: Arc<gpu::Context>, bins: &TwoHotBins, rows: usize) -> Self {
        assert!(rows > 0 && rows <= 65535);
        assert!(bins.len() <= u32::MAX as usize / rows);
        let support = gpu.create_buffer(gpu::BufferDesc {
            name: "categorical_support",
            size: (bins.len() * 4) as u64,
            memory: gpu::Memory::Shared,
        });
        // Host write precedes the first queue submission. Support is immutable
        // until the decoder has completed all uses before destruction.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bins.values().as_ptr(),
                support.data().cast(),
                bins.len(),
            );
        }
        let output = gpu.create_buffer(gpu::BufferDesc {
            name: "categorical_means",
            size: (3 * rows * 4) as u64,
            memory: gpu::Memory::Device,
        });
        let shader = gpu.create_shader(gpu::ShaderDesc {
            source: include_str!("categorical_gpu.wgsl"),
            naga_module: None,
        });
        let pipeline = gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            full_subgroup_size: None,
            name: "categorical_mean",
            data_layouts: &[&Bindings::layout()],
            compute: shader.at("main"),
        });
        Self {
            encoder: gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "categorical_means",
                buffer_count: 1,
                manual_barriers: false,
            }),
            pipeline,
            gpu,
            completion: None,
            bins: support,
            count: bins.len(),
            output,
            rows,
        }
    }

    /// Submitted producers and the returned regions remain live until the
    /// consumer's readback completes. Session updates are serialized by the core.
    pub fn decode(&mut self, sources: &[(&Session, usize); 3]) -> [gpu::BufferPiece; 3] {
        let mut regions = [self.output.at(0); 3];
        for (region, &(session, index)) in regions.iter_mut().zip(sources) {
            assert!(Arc::ptr_eq(&self.gpu, &session.context()));
            assert_eq!(
                session.slot_size(ExternalSlot::Output(index)),
                Some(self.rows * self.count * 4)
            );
            *region = session
                .output_buffer(index)
                .expect("known categorical output");
        }
        self.decode_into(&regions, self.output.at(0), 0);
        std::array::from_fn(|i| self.output.at((i * self.rows * 4) as u64))
    }

    fn decode_into(
        &mut self,
        sources: &[gpu::BufferPiece],
        output: gpu::BufferPiece,
        first: usize,
    ) {
        assert!(!sources.is_empty());
        let count = self
            .rows
            .checked_mul(sources.len())
            .and_then(|n| n.checked_add(first))
            .unwrap();
        assert!(count <= u32::MAX as usize);
        assert!(
            output
                .offset
                .checked_add((count * 4) as u64)
                .is_some_and(|end| end <= output.buffer.size())
        );
        for source in sources {
            assert_ne!(source.buffer, output.buffer);
            assert!(
                source
                    .offset
                    .checked_add((self.rows * self.count * 4) as u64)
                    .is_some_and(|end| end <= source.buffer.size())
            );
        }
        self.wait();
        self.encoder.start();
        {
            let mut pass = self.encoder.compute("categorical_means");
            let mut pipeline = pass.with(&self.pipeline);
            for (head, &logits) in sources.iter().enumerate() {
                pipeline.bind(
                    0,
                    &Bindings {
                        logits,
                        bins: self.bins.at(0),
                        means: output,
                        params: [
                            self.count as u32,
                            self.rows as u32,
                            (first + head * self.rows) as u32,
                            0,
                        ],
                    },
                );
                pipeline.dispatch([self.rows as u32, 1, 1]);
            }
        }
        self.completion = Some(self.gpu.submit(&mut self.encoder));
    }

    fn wait(&mut self) {
        if let Some(done) = self.completion.take() {
            assert!(
                self.gpu
                    .wait_for(&done, !0)
                    .expect("categorical mean completion")
            );
        }
    }
}

impl Drop for CategoricalGpu {
    fn drop(&mut self) {
        self.wait();
        self.gpu.destroy_compute_pipeline(&mut self.pipeline);
        self.gpu.destroy_command_encoder(&mut self.encoder);
        self.gpu.destroy_buffer(self.bins);
        self.gpu.destroy_buffer(self.output);
    }
}

#[cfg(test)]
pub(super) mod tests;
