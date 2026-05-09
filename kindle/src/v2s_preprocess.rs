//! GPU-side preprocess pass that turns the per-lane Dullahan-exported
//! BGRA8/RGBA8 frame ring into the `(N, 3, 192, 192)` f32 buffer V2-S
//! consumes — written *directly* into V2-S's `image` input slot via the
//! shared blade context, no fd/dmabuf interop and no DtoH roundtrip.
//!
//! Track B.2 in `docs/kindle_rl_pipeline_design.md`.  See the WGSL at
//! `src/shaders/v2s_preprocess.wgsl` for the per-pixel sampling math.
//!
//! # Lifecycle
//!
//! - One pipeline per `Agent`, built lazily on the first
//!   [`PreprocessPipeline::register_lane`] call.
//! - Per-lane source state holds the imported `Memory::External(Fd(...))`
//!   buffer plus the metadata needed to compute slot offsets each step.
//!   On `reset_lane` the producer-side fd is replaced; the host calls
//!   [`PreprocessPipeline::register_lane`] again with the new fd.
//! - [`PreprocessPipeline::dispatch_step`] enqueues N independent
//!   dispatches (one per registered lane) into a single command encoder
//!   and submits + waits.  Cheap because the workload is tiny
//!   (192/16 = 12 wgs × 12 wgs × N ≈ 576 wgs per step) and we'd block
//!   on V2-S's forward anyway.

use blade_graphics as bg;
use std::sync::Arc;

const DST_HW: u32 = 192;
/// `192 / 16` workgroups per axis — matches `@workgroup_size(16, 16, 1)`
/// in `v2s_preprocess.wgsl`.
const WORKGROUPS_PER_AXIS: u32 = 12;
/// Mirror of WGSL's `MAX_HUD_RECTS`.  Bumping this requires editing
/// the shader's `array<vec4<u32>, N>` declaration to match.
pub const MAX_HUD_RECTS: usize = 4;

/// Bindings struct laid out to match the WGSL's `var` names exactly.
/// Field names are load-bearing (blade-macros' `ShaderData` derive maps
/// them by name).
#[derive(blade_macros::ShaderData)]
struct PreprocessShaderData {
    src_words: bg::BufferPiece,
    dst: bg::BufferPiece,
    params: PreprocessParams,
    hud_rects: HudRects,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PreprocessParams {
    is_rgba: u32,
    lane: u32,
    src_w: u32,
    src_h: u32,
    dst_hw: u32,
    src_row_stride: u32,
    src_frame_words_offset: u32,
    hud_count: u32,
}

/// Inline uniform mirroring the WGSL `HudRects` struct: `MAX_HUD_RECTS`
/// `vec4<u32>` rects in dst-space pixel coords.  Per-rect WGSL alignment
/// is 16 bytes, which matches `[u32; 4]` here.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HudRects {
    rects: [[u32; 4]; MAX_HUD_RECTS],
}

/// Per-lane source-buffer record.  The blade `Buffer` is the imported
/// allocation; we also remember the SHM frame layout so each
/// `dispatch_step` can compute the right slot offset on the fly.
struct LaneSource {
    /// Imported Dullahan allocation, owned by this struct so blade
    /// keeps the fd alive.  Destroyed by [`PreprocessPipeline::drop`]
    /// or replaced by [`PreprocessPipeline::register_lane`].
    buffer: bg::Buffer,
    /// Bytes from start-of-allocation to the first frame slot (the SHM
    /// header + per-slot timestamp table sit before it).  Comes from
    /// the Dullahan `gpu_info` message.
    frame_data_offset_bytes: u64,
    /// Bytes per pixel-data frame (= `src_w * src_h * 4`).
    bytes_per_frame: u64,
    /// Width of one frame in pixels.
    src_w: u32,
    /// Height of one frame in pixels.
    src_h: u32,
    /// `1` if memory order is R-G-B-A (`VK_FORMAT_R8G8B8A8_UNORM`),
    /// `0` for B-G-R-A (the DXVK / opaque-mode default).
    is_rgba: u32,
}

/// Owns the V2-S preprocess compute pipeline + its per-lane source
/// records.  Built once at `Agent` construction and reused for the
/// agent's lifetime.
pub struct PreprocessPipeline {
    gpu: Arc<bg::Context>,
    pipeline: bg::ComputePipeline,
    encoder: bg::CommandEncoder,
    /// V2-S `image` input slot fetched from the meganeura session via
    /// `Session::input_buffer("image")`.  Lives for the agent's
    /// lifetime — the session owns the underlying buffer.
    dst_image: bg::BufferPiece,
    sources: Vec<Option<LaneSource>>,
    n_lanes: usize,
    /// HUD-mask rects in V2-S input (dst) pixel space, 0..192.  Set
    /// once per game via `set_hud_masks` and reused for every
    /// `dispatch_step`.  When empty, `hud_count == 0` and the shader's
    /// per-pixel mask loop is a no-op.
    hud_count: u32,
    hud_rects: [[u32; 4]; MAX_HUD_RECTS],
}

impl PreprocessPipeline {
    /// Build the compute pipeline and allocate the per-lane source
    /// slots.  `dst_image` is the `BufferPiece` returned by
    /// `meganeura::Session::input_buffer("image")` on the V2-S session;
    /// `n_lanes` is the agent's batch size.
    pub fn new(
        gpu: Arc<bg::Context>,
        dst_image: bg::BufferPiece,
        n_lanes: usize,
    ) -> Self {
        let shader_source = include_str!("shaders/v2s_preprocess.wgsl");
        let shader = gpu.create_shader(bg::ShaderDesc {
            source: shader_source,
            naga_module: None,
        });
        let layout = <PreprocessShaderData as bg::ShaderData>::layout();
        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: "v2s_preprocess",
            data_layouts: &[&layout],
            compute: shader.at("main"),
        });
        let encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "v2s_preprocess",
            // One frame in flight per dispatch + one queued at a time
            // is enough — the trainer's outer loop ping-pongs.
            buffer_count: 2,
        });
        Self {
            gpu,
            pipeline,
            encoder,
            dst_image,
            sources: (0..n_lanes).map(|_| None).collect(),
            n_lanes,
            hud_count: 0,
            hud_rects: [[0u32; 4]; MAX_HUD_RECTS],
        }
    }

    /// Configure the HUD-mask rects in V2-S input (192²) pixel space.
    /// Each rect is `(x0, y0, x1, y1)` with `x0 < x1`, `y0 < y1`, all
    /// in `[0, 192]`.  The shader zeros every output pixel inside any
    /// rect.  Pass an empty slice to disable masking; values past
    /// `MAX_HUD_RECTS` are silently dropped.
    pub fn set_hud_masks(&mut self, rects: &[(u32, u32, u32, u32)]) {
        self.hud_rects = [[0u32; 4]; MAX_HUD_RECTS];
        let n = rects.len().min(MAX_HUD_RECTS);
        for (i, (x0, y0, x1, y1)) in rects.iter().take(n).enumerate() {
            // Clamp + degenerate-rect guard so the shader's strict
            // `x < x1` / `y < y1` test never matches an empty rect.
            let x0c = (*x0).min(DST_HW);
            let y0c = (*y0).min(DST_HW);
            let x1c = (*x1).min(DST_HW).max(x0c);
            let y1c = (*y1).min(DST_HW).max(y0c);
            self.hud_rects[i] = [x0c, y0c, x1c, y1c];
        }
        self.hud_count = n as u32;
    }

    /// Import (or replace) the source buffer for one lane.
    ///
    /// The `fd` is the Dullahan-exported OPAQUE_FD owned by the producer
    /// (the same one the trainer's HIP path imported via
    /// `import_dullahan_fd`).  We import it into kindle's blade context
    /// here so the preprocess shader can sample from it directly — no
    /// fd duplication, no PCIe roundtrip.  Blade takes ownership of the
    /// fd via the import path and closes it when the buffer is destroyed.
    ///
    /// Re-call on `reset_lane` after the producer hands a fresh fd.
    pub fn register_lane(
        &mut self,
        lane: usize,
        fd: i32,
        allocation_size_bytes: u64,
        frame_data_offset_bytes: u64,
        bytes_per_frame: u64,
        src_w: u32,
        src_h: u32,
        is_rgba: bool,
    ) -> Result<(), String> {
        if lane >= self.n_lanes {
            return Err(format!(
                "lane {lane} out of range (n_lanes={})",
                self.n_lanes
            ));
        }
        let buffer = self.gpu.create_buffer(bg::BufferDesc {
            name: "v2s_preprocess_src",
            size: allocation_size_bytes,
            memory: bg::Memory::External(bg::ExternalMemorySource::Fd(Some(fd))),
        });
        if let Some(old) = self.sources[lane].take() {
            self.gpu.destroy_buffer(old.buffer);
        }
        self.sources[lane] = Some(LaneSource {
            buffer,
            frame_data_offset_bytes,
            bytes_per_frame,
            src_w,
            src_h,
            is_rgba: is_rgba as u32,
        });
        Ok(())
    }

    /// Drop the source for a lane (e.g. before re-registration with a
    /// brand-new fd).  Cheaper than letting `register_lane` chain the
    /// destroy when the caller wants explicit control over fd lifetime.
    pub fn release_lane(&mut self, lane: usize) {
        if lane < self.sources.len() {
            if let Some(old) = self.sources[lane].take() {
                self.gpu.destroy_buffer(old.buffer);
            }
        }
    }

    /// Run one preprocess dispatch per registered lane: read each
    /// lane's most-recently-presented frame slot and write it into the
    /// V2-S image buffer at `[lane, :, :, :]`.
    ///
    /// `slot_indices[lane]` is the SHM ring slot index the producer just
    /// wrote (matching what `VectorRustGameEnv::step_no_copy` already
    /// returns).  Lanes whose source isn't registered or whose slot is
    /// `None` are skipped.
    pub fn dispatch_step(&mut self, slot_indices: &[Option<u32>]) {
        debug_assert_eq!(slot_indices.len(), self.n_lanes);

        self.encoder.start();
        {
            let mut pass = self.encoder.compute("v2s_preprocess");
            let mut pc = pass.with(&self.pipeline);

            for (lane, slot) in slot_indices.iter().enumerate() {
                let Some(src) = self.sources[lane].as_ref() else {
                    continue;
                };
                let Some(slot_idx) = *slot else {
                    continue;
                };
                let frame_offset_bytes = src
                    .frame_data_offset_bytes
                    .saturating_add((slot_idx as u64) * src.bytes_per_frame);
                debug_assert!(frame_offset_bytes % 4 == 0);
                let src_frame_words_offset = (frame_offset_bytes / 4) as u32;

                pc.bind(
                    0,
                    &PreprocessShaderData {
                        src_words: src.buffer.into(),
                        dst: self.dst_image,
                        params: PreprocessParams {
                            is_rgba: src.is_rgba,
                            lane: lane as u32,
                            src_w: src.src_w,
                            src_h: src.src_h,
                            dst_hw: DST_HW,
                            src_row_stride: src.src_w,
                            src_frame_words_offset,
                            hud_count: self.hud_count,
                        },
                        hud_rects: HudRects {
                            rects: self.hud_rects,
                        },
                    },
                );
                pc.dispatch([WORKGROUPS_PER_AXIS, WORKGROUPS_PER_AXIS, 1]);
            }
        }
        let sp = self.gpu.submit(&mut self.encoder);
        let _ = self.gpu.wait_for(&sp, !0);
    }
}

impl Drop for PreprocessPipeline {
    fn drop(&mut self) {
        for slot in self.sources.iter_mut() {
            if let Some(src) = slot.take() {
                self.gpu.destroy_buffer(src.buffer);
            }
        }
        self.gpu.destroy_command_encoder(&mut self.encoder);
        self.gpu.destroy_compute_pipeline(&mut self.pipeline);
    }
}
