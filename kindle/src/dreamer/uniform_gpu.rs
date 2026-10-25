//! Exact ChaCha12 draws written into learner inputs on the shared GPU queue.
use std::sync::Arc;

use blade_graphics::{self as gpu, ShaderBindable as _, ShaderData as _};
use rand_chacha::ChaCha12Rng;

struct Bindings {
    dst: gpu::BufferPiece,
    params: [u32; 16],
}

impl gpu::ShaderData for Bindings {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("dst", gpu::ShaderBinding::Buffer),
                ("params", gpu::ShaderBinding::Plain { size: 64 }),
            ],
        }
    }

    fn fill(&self, mut context: gpu::PipelineContext) {
        self.dst.bind_to(&mut context, 0);
        self.params.bind_to(&mut context, 1);
    }
}

pub(super) struct UniformGpu {
    gpu: Arc<gpu::Context>,
    encoder: gpu::CommandEncoder,
    pipeline: gpu::ComputePipeline,
    completion: Option<gpu::SyncPoint>,
    #[cfg(test)]
    timing: bool,
}

impl UniformGpu {
    pub fn new(gpu: Arc<gpu::Context>) -> Self {
        let shader = gpu.create_shader(gpu::ShaderDesc {
            source: include_str!("uniform_gpu.wgsl"),
            naga_module: None,
        });
        let pipeline = gpu.create_compute_pipeline(gpu::ComputePipelineDesc { full_subgroup_size: None,
            name: "kindle_uniform_chacha12",
            data_layouts: &[&Bindings::layout()],
            compute: shader.at("main"),
        });
        Self {
            encoder: gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "kindle_uniforms",
                buffer_count: 1,
                manual_barriers: false,
            }),
            pipeline,
            gpu,
            completion: None,
            #[cfg(test)]
            timing: meganeura::GpuOptions::from_env().timing,
        }
    }

    /// Buffers belong to this context and remain live through completion.
    /// Targets must not overlap. Later session submissions on the shared queue
    /// consume the draws without a host wait; CPU writes must wait first.
    pub fn fill(&mut self, rng: &mut ChaCha12Rng, outputs: &[(gpu::BufferPiece, usize)]) {
        self.fill_impl(rng, outputs, false);
    }

    fn fill_impl(
        &mut self,
        rng: &mut ChaCha12Rng,
        outputs: &[(gpu::BufferPiece, usize)],
        raw_words: bool,
    ) {
        for (i, &(dst, len)) in outputs.iter().enumerate() {
            let bytes = len.checked_mul(4).expect("random output byte count");
            assert!(
                dst.offset
                    .checked_add(bytes as u64)
                    .is_some_and(|end| end <= dst.buffer.size())
            );
            assert!(len <= u32::MAX as usize - 15);
            assert!((len + 15).div_ceil(1024) <= 65535);
            for &(other, other_len) in &outputs[..i] {
                if dst.buffer == other.buffer && len != 0 && other_len != 0 {
                    assert!(
                        dst.offset + bytes as u64 <= other.offset
                            || other.offset + (other_len * 4) as u64 <= dst.offset
                    );
                }
            }
        }
        if outputs.iter().all(|o| o.1 == 0) {
            return;
        }
        self.wait();
        let mut params = [0; 16];
        for (word, bytes) in params[..8]
            .iter_mut()
            .zip(rng.get_seed().as_chunks::<4>().0)
        {
            *word = u32::from_le_bytes(*bytes);
        }
        let stream = rng.get_stream();
        params[10] = stream as u32;
        params[11] = (stream >> 32) as u32;
        params[14] = u32::from(raw_words);
        self.encoder.start();
        {
            let mut pass = self.encoder.compute("kindle_uniforms");
            let mut pipeline = pass.with(&self.pipeline);
            for &(dst, len) in outputs {
                if len == 0 {
                    continue;
                }
                let position = rng.get_word_pos();
                let block = (position >> 4) as u64;
                params[8] = block as u32;
                params[9] = (block >> 32) as u32;
                params[12] = len as u32;
                params[13] = (position & 15) as u32;
                pipeline.bind(0, &Bindings { dst, params });
                pipeline.dispatch([(params[12] + params[13]).div_ceil(16).div_ceil(64), 1, 1]);
                rng.set_word_pos(position + len as u128);
            }
        }
        self.completion = Some(self.gpu.submit(&mut self.encoder));
    }

    pub fn wait(&mut self) {
        if let Some(done) = self.completion.take() {
            assert!(
                self.gpu
                    .wait_for(&done, !0)
                    .expect("uniform GPU completion")
            );
            #[cfg(test)]
            if self.timing && self.gpu.capabilities().timing {
                let timings = self.encoder.last_timing();
                meganeura::profiler::record_gpu_timings(&meganeura::profiler::GpuTimings {
                    passes: timings
                        .passes
                        .iter()
                        .map(|&(name, at)| (name.to_owned(), at))
                        .collect(),
                    done: Some(timings.done),
                });
            }
        }
    }
}

impl Drop for UniformGpu {
    fn drop(&mut self) {
        self.wait();
        self.gpu.destroy_compute_pipeline(&mut self.pipeline);
        self.gpu.destroy_command_encoder(&mut self.encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};

    #[test]
    fn explicit_chacha_matches_pinned_std_rng() {
        for seed in [0, 1, 7, 701, 1009, u64::MAX] {
            let mut previous = StdRng::seed_from_u64(seed);
            let mut explicit = ChaCha12Rng::seed_from_u64(seed);
            for _ in 0..4097 {
                assert_eq!(previous.next_u32(), explicit.next_u32());
                assert_eq!(previous.next_u64(), explicit.next_u64());
                assert_eq!(
                    previous.random::<f32>().to_bits(),
                    explicit.random::<f32>().to_bits()
                );
            }
        }
    }

    #[test]
    fn uniform_shader_validates_without_a_gpu() {
        let module = naga::front::wgsl::parse_str(include_str!("uniform_gpu.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    #[ignore = "requires guarded GPU; exact ChaCha words, floats, tails and queue handoff"]
    fn gpu_uniforms_match_cpu_sequence() {
        const GUARD: usize = 64;
        const CANARY: u32 = 0x9abc1234;
        let gpu = meganeura::reference::gpu::shared_context();
        let device = gpu.device_information();
        assert_eq!(
            device.device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        assert!(!device.is_software_emulated);
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        let mut uniforms = UniformGpu::new(Arc::clone(&gpu));
        let mut checked_words = 0;
        // Include word/buffer boundaries, low-counter carry and full-period wrap.
        let cases = [
            (0, 0, 0),
            (1, 0, 0),
            (7, 1, 0),
            (15, 15, 1),
            (16, 16, u64::MAX),
            (17, 63, 0),
            (257, 65, 1 << 40),
            (4096, (1 << 36) - 17, u64::MAX),
            (129, (1 << 68) - 17, 17),
            (1_048_579, 7, 0x1234567812345678),
        ];
        for raw_words in [true, false] {
            for (case, (len, position, stream)) in cases.into_iter().enumerate() {
                let mut rng = ChaCha12Rng::seed_from_u64(701 + case as u64);
                rng.set_stream(stream);
                rng.set_word_pos(position);
                let mut reference = rng.clone();
                let buffer = gpu.create_buffer(gpu::BufferDesc {
                    name: "uniform_canaries",
                    size: ((len + 2 * GUARD) * 4) as u64,
                    memory: gpu::Memory::Shared,
                });
                unsafe {
                    std::slice::from_raw_parts_mut(buffer.data().cast::<u32>(), len + 2 * GUARD)
                        .fill(CANARY);
                }
                uniforms.fill_impl(&mut rng, &[(buffer.at((GUARD * 4) as u64), len)], raw_words);
                uniforms.wait();
                let values = unsafe {
                    std::slice::from_raw_parts(buffer.data().cast::<u32>(), len + 2 * GUARD)
                };
                assert!(
                    values[..GUARD]
                        .iter()
                        .chain(&values[GUARD + len..])
                        .all(|&x| x == CANARY)
                );
                for (index, &actual) in values[GUARD..GUARD + len].iter().enumerate() {
                    let expected = if raw_words {
                        reference.next_u32()
                    } else {
                        reference.random::<f32>().to_bits()
                    };
                    assert_eq!(
                        actual, expected,
                        "case {case}, raw {raw_words}, index {index}"
                    );
                }
                assert_eq!(rng.get_word_pos(), reference.get_word_pos());
                assert_eq!(rng.get_stream(), reference.get_stream());
                assert_eq!(rng.get_seed(), reference.get_seed());
                assert_eq!(rng.next_u64(), reference.next_u64());
                checked_words += len;
                gpu.destroy_buffer(buffer);
            }
        }
        // Consecutive submissions and multi-output fills share one logical stream.
        // Each separate output begins on an aligned binding, not a block boundary.
        let buffer = gpu.create_buffer(gpu::BufferDesc {
            name: "uniform_split_canaries",
            size: 4 * 256,
            memory: gpu::Memory::Shared,
        });
        unsafe {
            std::slice::from_raw_parts_mut(buffer.data().cast::<u32>(), 256).fill(CANARY);
        }
        let mut rng = ChaCha12Rng::seed_from_u64(701);
        rng.set_word_pos(13);
        let mut reference = rng.clone();
        uniforms.fill(
            &mut rng,
            &[
                (buffer.at(0), 17),
                (buffer.at(256), 0),
                (buffer.at(256), 31),
            ],
        );
        uniforms.fill(&mut rng, &[(buffer.at(512), 65)]);
        uniforms.wait();
        let values = unsafe { std::slice::from_raw_parts(buffer.data().cast::<u32>(), 256) };
        for (index, &actual) in values.iter().enumerate() {
            let expected = if (0..17).contains(&index)
                || (64..95).contains(&index)
                || (128..193).contains(&index)
            {
                checked_words += 1;
                reference.random::<f32>().to_bits()
            } else {
                CANARY
            };
            assert_eq!(actual, expected, "split output {index}");
        }
        assert_eq!(rng.get_word_pos(), reference.get_word_pos());
        gpu.destroy_buffer(buffer);

        // The consumer waits only for its own submission, not the RNG producer.
        let mut graph = meganeura::Graph::new();
        let input = graph.input("uniforms", &[257]);
        let output = graph.neg(input);
        graph.set_outputs(vec![output]);
        let mut session =
            super::super::runtime::build_session(&graph, &gpu, meganeura::Mode::Inference, false);
        for _ in 0..3 {
            uniforms.fill(
                &mut rng,
                &[(session.input_buffer("uniforms").unwrap(), 257)],
            );
            session.step();
            let mut actual = vec![0.0; 257];
            session.wait_read_output(0, &mut actual);
            for value in actual {
                assert_eq!(value.to_bits(), (-reference.random::<f32>()).to_bits());
                checked_words += 1;
            }
        }
        uniforms.wait();
        assert_eq!(rng.get_word_pos(), reference.get_word_pos());
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        println!(
            "uniform_gpu_exact_words={checked_words}; cases=20; split_fills=2; queue_handoffs=3"
        );
    }
}
