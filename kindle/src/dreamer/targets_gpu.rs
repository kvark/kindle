//! Detached learner targets on the learner's queue. No model/optimizer changes.
use std::sync::Arc;

use blade_graphics::{self as gpu, ShaderBindable as _, ShaderData as _};
use meganeura::{Session, runtime::ExternalSlot};

use super::{DreamerConfig, distributions::TwoHotBins, replay::SequenceBatch, world};

type Params = [u32; 16];
type CopyRegion = (gpu::BufferPiece, gpu::BufferPiece, usize);

macro_rules! bindings {
    ($name:ident { $($field:ident),+ }) => {
        struct $name { $( $field: gpu::BufferPiece, )+ params: Params }
        impl gpu::ShaderData for $name {
            fn layout() -> gpu::ShaderDataLayout {
                gpu::ShaderDataLayout { bindings: vec![
                    $( (stringify!($field), gpu::ShaderBinding::Buffer), )+
                    ("params", gpu::ShaderBinding::Plain { size: 64 }),
                ] }
            }
            fn fill(&self, mut context: gpu::PipelineContext) {
                let mut index = 0;
                $( self.$field.bind_to(&mut context, index); index += 1; )+
                self.params.bind_to(&mut context, index);
            }
        }
    }
}
bindings!(ReturnsData {
    rewards,
    values,
    continuation,
    bonus,
    replay_bonus,
    replay,
    returns,
    weights,
    replay_returns,
    replay_weights,
    continuation_target,
    world_weight,
    status
});
bindings!(NormalizeData { returns, state });
bindings!(EncodeData {
    values,
    bins,
    encoded,
    status
});
bindings!(ActorData {
    returns,
    values,
    weights,
    actions,
    state,
    encoded,
    status
});
bindings!(ReportData {
    rewards,
    values,
    continuation,
    bonus,
    replay_bonus,
    replay,
    returns,
    weights,
    state,
    report,
    status
});

pub(super) struct TargetsGpu {
    gpu: Arc<gpu::Context>,
    config: DreamerConfig,
    encoder: gpu::CommandEncoder,
    pipelines: [gpu::ComputePipeline; 6],
    completion: Option<gpu::SyncPoint>,
    pending: bool,
    params: Params,
    returns: gpu::Buffer,
    replay_returns: gpu::Buffer,
    reward_target: gpu::Buffer,
    continuation_target: gpu::Buffer,
    world_weight: gpu::Buffer,
    replay: gpu::Buffer,
    support: gpu::Buffer,
    state: gpu::Buffer,
    report: gpu::Buffer,
    status: gpu::Buffer,
    zero: gpu::Buffer,
}

fn check_region(region: gpu::BufferPiece, elements: usize) {
    assert_eq!(region.offset % 4, 0);
    let bytes = elements.checked_mul(4).unwrap();
    assert!(
        region
            .offset
            .checked_add(bytes as u64)
            .is_some_and(|end| end <= region.buffer.size())
    );
}

fn input(session: &Session, name: &str, elements: usize) -> gpu::BufferPiece {
    assert_eq!(
        session.slot_size(ExternalSlot::Input(name)),
        Some(elements * 4),
        "{name}"
    );
    let region = session.input_buffer(name).expect("known F32 target input");
    check_region(region, elements);
    region
}

fn output(session: &Session, index: usize, elements: usize) -> gpu::BufferPiece {
    assert_eq!(
        session.slot_size(ExternalSlot::Output(index)),
        Some(elements * 4)
    );
    let region = session
        .output_buffer(index)
        .expect("known F32 target producer");
    check_region(region, elements);
    region
}

fn write_shared(buffer: gpu::Buffer, values: &[f32]) {
    assert_eq!(buffer.size(), (values.len() * 4) as u64);
    // Only initialization, restore, or a host batch after the last consumer.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.data().cast(), values.len());
    }
}

impl TargetsGpu {
    pub fn new(gpu: Arc<gpu::Context>, config: &DreamerConfig, bins: &TwoHotBins) -> Self {
        let starts = config.batch_size.checked_mul(config.batch_length).unwrap();
        let rows = starts.checked_mul(config.imagination_length).unwrap();
        assert!(starts > 0 && rows > 0 && config.batch_length >= 2);
        assert!(
            rows.next_power_of_two() <= 8192,
            "resident percentile capacity is 8192 returns"
        );
        assert_eq!(config.world_backprop_length, config.batch_length);
        assert_eq!(config.world_microbatch_size(), config.batch_size);
        assert_eq!(bins.len(), config.value_bins);
        assert!(bins.values().iter().all(|value| value.is_finite()));
        assert!(bins.values().windows(2).all(|pair| pair[0] < pair[1]));
        let alloc = |name, count, memory| {
            gpu.create_buffer(gpu::BufferDesc {
                name,
                size: (count * 4) as u64,
                memory,
            })
        };
        let support = alloc("target_bins", bins.len(), gpu::Memory::Shared);
        write_shared(support, bins.values());
        let state = alloc("target_normalizer", 2, gpu::Memory::Shared);
        write_shared(state, &[0.0, 0.0]);
        let zero = alloc(
            "target_zeros",
            config.batch_size * config.value_bins,
            gpu::Memory::Shared,
        );
        write_shared(zero, &vec![0.0; config.batch_size * config.value_bins]);
        let pipeline = |source: &str, entry: &str, layout: gpu::ShaderDataLayout| {
            let shader = gpu.create_shader(gpu::ShaderDesc {
                source,
                naga_module: None,
            });
            gpu.create_compute_pipeline(gpu::ComputePipelineDesc { full_subgroup_size: None,
                name: entry,
                data_layouts: &[&layout],
                compute: shader.at(entry),
            })
        };
        let normalization = include_str!("targets_normalize.wgsl")
            .replace("_SORT_LENGTH_", &rows.next_power_of_two().to_string());
        let pipelines = [
            pipeline(
                include_str!("targets_returns.wgsl"),
                "imagined",
                ReturnsData::layout(),
            ),
            pipeline(&normalization, "normalize", NormalizeData::layout()),
            pipeline(
                include_str!("targets_returns.wgsl"),
                "replayed",
                ReturnsData::layout(),
            ),
            pipeline(
                include_str!("targets_actor.wgsl"),
                "actor",
                ActorData::layout(),
            ),
            pipeline(
                include_str!("targets_encode.wgsl"),
                "encode",
                EncodeData::layout(),
            ),
            pipeline(
                include_str!("targets_report.wgsl"),
                "summarize",
                ReportData::layout(),
            ),
        ];
        Self {
            config: config.clone(),
            encoder: gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "learner_targets",
                buffer_count: 1,
                manual_barriers: false,
            }),
            pipelines,
            completion: None,
            pending: false,
            params: [
                starts as u32,
                config.imagination_length as u32,
                config.batch_size as u32,
                config.batch_length as u32,
                config.value_bins as u32,
                config.action_count as u32,
                u32::from(config.uses_disagreement()),
                0,
                config.lambda.to_bits(),
                config.continuation_discount().to_bits(),
                config.intrinsic_reward_scale.to_bits(),
                config.return_norm_rate.to_bits(),
                0,
                0,
                0,
                0,
            ],
            returns: alloc("target_returns", rows, gpu::Memory::Device),
            replay_returns: alloc(
                "target_replay_returns",
                starts - config.batch_size,
                gpu::Memory::Device,
            ),
            reward_target: alloc("target_rewards", starts * bins.len(), gpu::Memory::Device),
            continuation_target: alloc("target_continuations", starts, gpu::Memory::Device),
            world_weight: alloc("target_world_weights", starts, gpu::Memory::Device),
            replay: alloc("target_replay_inputs", starts * 3, gpu::Memory::Shared),
            support,
            state,
            zero,
            report: alloc("target_report", 25, gpu::Memory::Shared),
            status: alloc("target_status", 1, gpu::Memory::Shared),
            gpu,
        }
    }

    /// Submitted means/imagination/posterior are on this same queue. Their
    /// buffers and the behavior inputs remain pinned through both training waits.
    pub fn generate(
        &mut self,
        batch: &SequenceBatch,
        imagination: &Session,
        posterior: (&Session, usize),
        behavior: &Session,
        means: [gpu::BufferPiece; 3],
    ) {
        assert!(
            !self.pending,
            "previous learner consumers have not completed"
        );
        for session in [imagination, posterior.0, behavior] {
            assert!(Arc::ptr_eq(&self.gpu, &session.context()));
        }
        self.wait();
        let starts = self.params[0] as usize;
        let rows = starts * self.config.imagination_length;
        let replay_rows = starts - self.config.batch_size;
        let all_rows = rows + starts;
        for region in means {
            check_region(region, all_rows);
        }
        assert_eq!(batch.rewards.len(), self.config.batch_length);
        assert_eq!(batch.flags.len(), self.config.batch_length);
        // This is the existing CPU replay's scalar boundary, not generated targets.
        let mut replay = vec![0.0; starts * 3];
        for t in 0..self.config.batch_length {
            assert_eq!(batch.rewards[t].len(), self.config.batch_size);
            assert_eq!(batch.flags[t].len(), self.config.batch_size);
            for b in 0..self.config.batch_size {
                let i = t * self.config.batch_size + b;
                replay[i] = batch.rewards[t][b];
                replay[starts + i] = f32::from(batch.flags[t][b].is_last);
                replay[2 * starts + i] = f32::from(batch.flags[t][b].is_terminal);
            }
        }
        write_shared(self.replay, &replay);
        write_shared(self.status, &[0.0]);
        let weights = input(behavior, "imagined_weight", rows);
        let continuation = output(imagination, world::IMAGINATION_CONTINUATION, all_rows);
        let (bonus, replay_bonus) = if self.config.uses_disagreement() {
            (
                output(imagination, world::IMAGINATION_BONUS, rows),
                output(posterior.0, posterior.1, starts),
            )
        } else {
            (self.zero.at(0), self.zero.at(0))
        };
        let data = ReturnsData {
            rewards: means[0],
            values: means[1],
            continuation,
            bonus,
            replay_bonus,
            replay: self.replay.at(0),
            returns: self.returns.at(0),
            weights,
            replay_returns: self.replay_returns.at(0),
            replay_weights: input(behavior, "replay_weight", replay_rows),
            continuation_target: self.continuation_target.at(0),
            world_weight: self.world_weight.at(0),
            status: self.status.at(0),
            params: self.params,
        };
        self.encoder.start();
        // Separate automatic-barrier passes are intentional: every consumer
        // observes the preceding complete producer, including the EMA update.
        {
            let mut pass = self.encoder.compute("target_returns");
            let mut pipe = pass.with(&self.pipelines[0]);
            pipe.bind(0, &data);
            pipe.dispatch([(starts as u32).div_ceil(64), 1, 1]);
        }
        {
            let mut pass = self.encoder.compute("target_normalizer");
            let mut pipe = pass.with(&self.pipelines[1]);
            pipe.bind(
                0,
                &NormalizeData {
                    returns: self.returns.at(0),
                    state: self.state.at(0),
                    params: self.params,
                },
            );
            pipe.dispatch([1, 1, 1]);
        }
        {
            let mut pass = self.encoder.compute("target_replay_returns");
            let mut pipe = pass.with(&self.pipelines[2]);
            pipe.bind(0, &data);
            pipe.dispatch([(self.config.batch_size as u32).div_ceil(64), 1, 1]);
        }
        {
            let mut pass = self.encoder.compute("target_actions");
            let mut pipe = pass.with(&self.pipelines[3]);
            pipe.bind(
                0,
                &ActorData {
                    returns: self.returns.at(0),
                    values: means[1],
                    weights,
                    actions: output(
                        imagination,
                        world::IMAGINATION_ACTION,
                        rows * self.config.action_count,
                    ),
                    state: self.state.at(0),
                    encoded: input(behavior, "action_target", rows * self.config.action_count),
                    status: self.status.at(0),
                    params: self.params,
                },
            );
            pipe.dispatch([(rows as u32).div_ceil(64), 1, 1]);
        }
        {
            let mut pass = self.encoder.compute("target_two_hot");
            let mut pipe = pass.with(&self.pipelines[4]);
            for (values, target, count) in [
                (
                    self.returns.at(0),
                    input(
                        behavior,
                        "imagined_value_target",
                        rows * self.config.value_bins,
                    ),
                    rows,
                ),
                (
                    means[2],
                    input(
                        behavior,
                        "imagined_slow_target",
                        rows * self.config.value_bins,
                    ),
                    rows,
                ),
                (
                    self.replay_returns.at(0),
                    input(
                        behavior,
                        "replay_value_target",
                        replay_rows * self.config.value_bins,
                    ),
                    replay_rows,
                ),
                (
                    means[2],
                    input(
                        behavior,
                        "replay_slow_target",
                        replay_rows * self.config.value_bins,
                    ),
                    replay_rows,
                ),
                (self.replay.at(0), self.reward_target.at(0), starts),
            ] {
                let mut params = self.params;
                params[12] = count as u32;
                pipe.bind(
                    0,
                    &EncodeData {
                        values,
                        bins: self.support.at(0),
                        encoded: target,
                        status: self.status.at(0),
                        params,
                    },
                );
                pipe.dispatch([count as u32, 1, 1]);
            }
        }
        {
            let mut pass = self.encoder.compute("target_report");
            let mut pipe = pass.with(&self.pipelines[5]);
            pipe.bind(
                0,
                &ReportData {
                    rewards: means[0],
                    values: means[1],
                    continuation,
                    bonus,
                    replay_bonus,
                    replay: self.replay.at(0),
                    returns: self.returns.at(0),
                    weights,
                    state: self.state.at(0),
                    report: self.report.at(0),
                    status: self.status.at(0),
                    params: self.params,
                },
            );
            pipe.dispatch([1, 1, 1]);
        }
        self.completion = Some(self.gpu.submit(&mut self.encoder));
        self.pending = true;
    }

    /// Source regions remain owned until DeviceCopies and both learner consumers
    /// finish. World late targets do not overwrite a prefix input or activation.
    pub fn handoffs(
        &self,
        world: &Session,
        behavior: &Session,
        imagination: &Session,
    ) -> Vec<CopyRegion> {
        assert!(self.pending);
        for session in [world, behavior, imagination] {
            assert!(Arc::ptr_eq(&self.gpu, &session.context()));
        }
        let rows = self.config.batch_size;
        let length = self.config.batch_length;
        let bins = self.config.value_bins;
        let starts = rows * length;
        let mut regions = Vec::new();
        let mut add = |source: gpu::BufferPiece, offset: usize, target, elements| {
            let source = gpu::BufferPiece {
                offset: source.offset.checked_add((offset * 4) as u64).unwrap(),
                ..source
            };
            check_region(source, elements);
            check_region(target, elements);
            let bytes = elements * 4;
            assert!(
                source.buffer != target.buffer
                    || source.offset + bytes as u64 <= target.offset
                    || target.offset + bytes as u64 <= source.offset,
                "overlapping target handoff"
            );
            regions.push((source, target, bytes));
        };
        let features = output(
            imagination,
            world::IMAGINATION_FEATURE,
            starts * self.config.imagination_length * self.config.feature_dim(),
        );
        let feature_elements = (starts - rows) * self.config.feature_dim();
        add(
            features,
            0,
            input(behavior, "replay_feature", feature_elements),
            feature_elements,
        );
        for t in 0..length {
            for (name, source, width) in [
                ("reward_target", self.reward_target.at(0), bins),
                ("continuation_target", self.continuation_target.at(0), 1),
                ("replay_value_weight", self.world_weight.at(0), 1),
            ] {
                add(
                    source,
                    t * rows * width,
                    input(world, &format!("{name}_{t}"), rows * width),
                    rows * width,
                );
            }
            for name in ["replay_value_target", "replay_slow_target"] {
                let (source, offset) = if t + 1 < length {
                    (
                        input(behavior, name, (starts - rows) * bins),
                        t * rows * bins,
                    )
                } else {
                    (self.zero.at(0), 0)
                };
                add(
                    source,
                    offset,
                    input(world, &format!("{name}_{t}"), rows * bins),
                    rows * bins,
                );
            }
            if self.config.actor_critic_gradient {
                for (name, source, width) in [
                    (
                        "initial_action_target",
                        "action_target",
                        self.config.action_count,
                    ),
                    ("initial_weight", "imagined_weight", 1),
                    ("initial_value_target", "imagined_value_target", bins),
                    ("initial_slow_target", "imagined_slow_target", bins),
                ] {
                    add(
                        input(
                            behavior,
                            source,
                            starts * self.config.imagination_length * width,
                        ),
                        t * rows * width,
                        input(world, &format!("{name}_{t}"), rows * width),
                        rows * width,
                    );
                }
            }
        }
        regions
    }

    pub fn report(&mut self) -> [f32; 25] {
        assert!(self.pending);
        self.wait();
        // Host-coherent Shared memory is read only after its writer completes.
        let status = unsafe { self.status.data().cast::<u32>().read() };
        assert_eq!(status, 0, "invalid resident learner targets");
        let report = unsafe { self.report.data().cast::<[f32; 25]>().read() };
        assert!(report.iter().all(|value| value.is_finite()));
        report
    }

    pub fn normalizer_state(&mut self) -> [f32; 2] {
        self.wait();
        let state = unsafe { self.state.data().cast::<[f32; 2]>().read() };
        assert!(state.iter().all(|value| value.is_finite()));
        state
    }

    pub fn restore_normalizer(&mut self, state: [f32; 2]) {
        assert!(!self.pending);
        assert!(state.iter().all(|value| value.is_finite()));
        self.wait();
        write_shared(self.state, &state);
    }

    /// Called only after world training, all device handoffs and behavior training
    /// have waited. A generator-only fence cannot authorize input-buffer reuse.
    pub fn finish_consumers(&mut self) {
        assert!(self.pending);
        self.wait();
        self.pending = false;
    }

    fn wait(&mut self) {
        if let Some(completion) = self.completion.take() {
            assert!(
                self.gpu
                    .wait_for(&completion, !0)
                    .expect("learner target completion")
            );
        }
    }
}

impl Drop for TargetsGpu {
    fn drop(&mut self) {
        self.wait();
        for pipeline in &mut self.pipelines {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        self.gpu.destroy_command_encoder(&mut self.encoder);
        for buffer in [
            self.returns,
            self.replay_returns,
            self.reward_target,
            self.continuation_target,
            self.world_weight,
            self.replay,
            self.support,
            self.state,
            self.report,
            self.status,
            self.zero,
        ] {
            self.gpu.destroy_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests;
