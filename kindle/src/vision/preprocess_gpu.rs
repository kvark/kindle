//! GPU letterboxing, RGB normalization and channel-major patch packing.
//! CPU images upload their original bytes; resident images use the same kernel.
//! JEPA interpolation stays in F32. The RGB64 control matches Pillow's
//! antialiased bilinear filter and per-axis RGB8 rounding on the GPU.
//! Padding is exactly zero in normalized space.

use std::{collections::HashMap, sync::Arc};

use blade_graphics::{self as gpu, ShaderBindable as _, ShaderData as _};
use meganeura::{Session, runtime::ExternalSlot};

mod rgb_filter;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Bgra8,
}

impl PixelFormat {
    fn channels(self) -> u32 {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 | Self::Bgra8 => 4,
        }
    }
}

/// Row stride is in bytes. Alpha is ignored, never used to premultiply RGB.
#[derive(Clone, Copy, Debug)]
pub struct FrameLayout {
    width: u32,
    height: u32,
    row_stride: u32,
    format: PixelFormat,
}

impl FrameLayout {
    pub fn new(width: u32, height: u32, row_stride: u32, format: PixelFormat) -> Self {
        assert!(width > 0 && height > 0, "empty frame");
        let row_bytes = width
            .checked_mul(format.channels())
            .expect("frame width overflow");
        assert!(row_stride >= row_bytes, "frame stride is too short");
        let layout = Self {
            width,
            height,
            row_stride,
            format,
        };
        layout.byte_len();
        layout
    }

    fn byte_len(self) -> u32 {
        (self.height - 1)
            .checked_mul(self.row_stride)
            .and_then(|bytes| bytes.checked_add(self.width * self.format.channels()))
            .expect("frame exceeds GPU address range")
    }
}

/// Borrowed packed GPU pixels, including a frame inside a capture ring buffer.
/// This does not import a capture allocation or manage its producer handshake.
#[derive(Clone, Copy)]
pub struct GpuFrame<'a> {
    gpu: &'a Arc<gpu::Context>,
    buffer: &'a gpu::Buffer,
    offset: u32,
    layout: FrameLayout,
}

impl<'a> GpuFrame<'a> {
    /// # Safety
    /// `buffer` must belong to `context` and remain allocated and unmodified
    /// until encoding returns. Producer writes must be complete or submitted
    /// earlier on this context's queue with the necessary memory dependencies.
    /// An external capture also needs its ownership/semaphore handshake; a
    /// shared-memory ready flag alone is not a GPU memory barrier.
    pub unsafe fn from_buffer(
        context: &'a Arc<gpu::Context>,
        buffer: &'a gpu::Buffer,
        offset: u64,
        layout: FrameLayout,
    ) -> Self {
        let end = offset
            .checked_add(u64::from(layout.byte_len()))
            .expect("frame offset overflow");
        assert!(
            end <= u64::from(u32::MAX),
            "frame exceeds GPU address range"
        );
        assert!(
            end.next_multiple_of(4) <= buffer.size(),
            "frame exceeds buffer"
        );
        Self {
            gpu: context,
            buffer,
            offset: offset as u32,
            layout,
        }
    }
}

struct Bindings {
    pixels: gpu::BufferPiece,
    patches: gpu::BufferPiece,
    filter: gpu::BufferPiece,
    params: [u32; 16],
}

impl gpu::ShaderData for Bindings {
    fn layout() -> gpu::ShaderDataLayout {
        gpu::ShaderDataLayout {
            bindings: vec![
                ("pixels", gpu::ShaderBinding::Buffer),
                ("patches", gpu::ShaderBinding::Buffer),
                ("rgb_coefficients", gpu::ShaderBinding::Buffer),
                ("params", gpu::ShaderBinding::Plain { size: 64 }),
            ],
        }
    }

    fn fill(&self, mut context: gpu::PipelineContext) {
        self.pixels.bind_to(&mut context, 0);
        self.patches.bind_to(&mut context, 1);
        self.filter.bind_to(&mut context, 2);
        self.params.bind_to(&mut context, 3);
    }
}

pub(crate) struct GpuPreprocessor {
    gpu: Arc<gpu::Context>,
    pipeline: gpu::ComputePipeline,
    encoder: gpu::CommandEncoder,
    completion: Option<gpu::SyncPoint>,
    upload: Option<gpu::Buffer>,
    rgb_filters: HashMap<(u32, u32), gpu::Buffer>,
    streams: usize,
    size: u32,
    patch: u32,
    input: &'static str,
    centered_rgb: bool,
}

impl GpuPreprocessor {
    pub fn new(gpu: Arc<gpu::Context>, streams: usize, size: usize, patch: usize) -> Self {
        validate_shape(streams, size, patch);
        let shader = gpu.create_shader(gpu::ShaderDesc {
            source: include_str!("preprocess.wgsl"),
            naga_module: None,
        });
        let pipeline = gpu.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "kindle_pixels_to_patches",
            data_layouts: &[&Bindings::layout()],
            compute: shader.at("main"),
        });
        let encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
            name: "kindle_pixel_preprocessing",
            buffer_count: 1,
            manual_barriers: false,
        });
        Self {
            gpu,
            pipeline,
            encoder,
            completion: None,
            upload: None,
            rgb_filters: HashMap::new(),
            streams,
            size: size.try_into().expect("image size overflow"),
            patch: patch.try_into().expect("patch size overflow"),
            input: "patches",
            centered_rgb: false,
        }
    }

    /// Learned RGB control: one antialiased full-frame resize, channel-major values in
    /// [-0.5, 0.5]. No letterbox, ImageNet statistics or second upscale.
    pub fn rgb64(gpu: Arc<gpu::Context>, streams: usize) -> Self {
        let mut pixels = Self::new(gpu, streams, 64, 64);
        pixels.input = "observation";
        pixels.centered_rgb = true;
        pixels
    }

    pub fn cpu_frames(&mut self, session: &mut Session, frames: &[(usize, &[u8], usize, usize)]) {
        let mut bytes = 0_u32;
        let layouts: Vec<_> = frames
            .iter()
            .map(|&(stream, pixels, width, height)| {
                let width: u32 = width.try_into().expect("frame width overflow");
                let height = height.try_into().expect("frame height overflow");
                let layout = FrameLayout::new(
                    width,
                    height,
                    width.checked_mul(3).expect("frame width overflow"),
                    PixelFormat::Rgb8,
                );
                assert_eq!(pixels.len(), layout.byte_len() as usize);
                let offset = bytes;
                bytes = bytes
                    .checked_add(padded_size(layout.byte_len()))
                    .expect("frame batch overflow");
                (stream, offset, layout)
            })
            .collect();
        validate_streams(layouts.iter().map(|row| row.0), self.streams);
        if frames.is_empty() {
            return;
        }
        self.wait();
        if self
            .upload
            .is_none_or(|buffer| buffer.size() < u64::from(bytes))
        {
            if let Some(buffer) = self.upload.take() {
                self.gpu.destroy_buffer(buffer);
            }
            self.upload = Some(self.gpu.create_buffer(gpu::BufferDesc {
                name: "kindle_rgb_upload",
                size: u64::from(bytes),
                memory: gpu::Memory::Shared,
            }));
        }
        let upload = self.upload.unwrap();
        for ((_, pixels, _, _), &(_, offset, layout)) in frames.iter().zip(&layouts) {
            // This owned, host-visible staging allocation is idle after wait().
            unsafe {
                let target = upload.data().add(offset as usize);
                std::ptr::copy_nonoverlapping(pixels.as_ptr(), target, pixels.len());
                std::ptr::write_bytes(
                    target.add(pixels.len()),
                    0,
                    padded_size(layout.byte_len()) as usize - pixels.len(),
                );
            }
        }
        let target = self.target(session);
        let bindings = layouts
            .iter()
            .map(|&(stream, offset, layout)| {
                self.bindings(upload.into(), target, stream, offset, layout)
            })
            .collect::<Vec<_>>();
        self.submit(target, &bindings);
    }

    pub fn gpu_frames(&mut self, session: &mut Session, frames: &[(usize, GpuFrame<'_>)]) {
        validate_streams(frames.iter().map(|row| row.0), self.streams);
        for (_, frame) in frames {
            assert!(Arc::ptr_eq(&self.gpu, frame.gpu), "different GPU context");
        }
        if frames.is_empty() {
            return;
        }
        let target = self.target(session);
        let bindings = frames
            .iter()
            .map(|&(stream, frame)| {
                self.bindings(
                    (*frame.buffer).into(),
                    target,
                    stream,
                    frame.offset,
                    frame.layout,
                )
            })
            .collect::<Vec<_>>();
        self.submit(target, &bindings);
    }

    fn elements(&self) -> u32 {
        3 * self.size * self.size
    }

    fn target(&self, session: &mut Session) -> gpu::BufferPiece {
        assert!(
            Arc::ptr_eq(&self.gpu, &session.context()),
            "different GPU context"
        );
        assert_eq!(
            session.slot_size(ExternalSlot::Input(self.input)),
            Some(self.streams * self.elements() as usize * size_of::<f32>())
        );
        session.wait();
        session
            .input_buffer(self.input)
            .expect("encoder patches input")
    }

    fn bindings(
        &mut self,
        pixels: gpu::BufferPiece,
        patches: gpu::BufferPiece,
        stream: usize,
        offset: u32,
        layout: FrameLayout,
    ) -> Bindings {
        let [width, height, x, y] = if self.centered_rgb {
            [self.size as usize, self.size as usize, 0, 0]
        } else {
            super::preprocess::letterbox_geometry(
                layout.width as usize,
                layout.height as usize,
                self.size as usize,
            )
        };
        // Geometry coefficients only; image data never crosses the host boundary.
        let filter = if self.centered_rgb {
            (*self
                .rgb_filters
                .entry((layout.width, layout.height))
                .or_insert_with(|| {
                    let coefficients = rgb_filter::coefficients(layout.width, layout.height);
                    let buffer = self.gpu.create_buffer(gpu::BufferDesc {
                        name: "rgb64_filter",
                        size: (coefficients.len() * size_of::<u32>()) as u64,
                        memory: gpu::Memory::Shared,
                    });
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            coefficients.as_ptr().cast::<u8>(),
                            buffer.data(),
                            coefficients.len() * size_of::<u32>(),
                        );
                    }
                    buffer
                }))
            .into()
        } else {
            pixels
        };
        Bindings {
            pixels,
            patches,
            filter,
            params: [
                layout.width,
                layout.height,
                layout.row_stride,
                layout.format.channels(),
                offset,
                u32::from(layout.format == PixelFormat::Bgra8),
                self.size,
                self.patch,
                width as u32,
                height as u32,
                x as u32,
                y as u32,
                u32::try_from(stream)
                    .unwrap()
                    .checked_mul(self.elements())
                    .expect("patch offset overflow"),
                u32::from(self.centered_rgb),
                0,
                0,
            ],
        }
    }

    fn submit(&mut self, target: gpu::BufferPiece, bindings: &[Bindings]) {
        self.wait();
        self.encoder.start();
        if bindings.len() != self.streams {
            let bytes = self.streams as u64 * u64::from(self.elements()) * 4;
            self.encoder
                .transfer("clear_inactive_patches")
                .fill_buffer(target, bytes, 0);
        }
        {
            let mut pass = self.encoder.compute("pixels_to_patches");
            let mut dispatch = pass.with(&self.pipeline);
            for binding in bindings {
                dispatch.bind(0, binding);
                dispatch.dispatch([(self.size * self.size).div_ceil(64), 1, 1]);
            }
        }
        self.completion = Some(self.gpu.submit(&mut self.encoder));
        // Blade's same-queue submission barriers order writes before the encoder.
        // No readback or host wait is needed before Session::step().
    }

    fn wait(&mut self) {
        if let Some(completion) = self.completion.take() {
            assert!(
                self.gpu
                    .wait_for(&completion, !0)
                    .expect("pixel preprocessing wait failed")
            );
        }
    }
}

impl Drop for GpuPreprocessor {
    fn drop(&mut self) {
        self.wait();
        if let Some(buffer) = self.upload.take() {
            self.gpu.destroy_buffer(buffer);
        }
        for (_, buffer) in self.rgb_filters.drain() {
            self.gpu.destroy_buffer(buffer);
        }
        self.gpu.destroy_compute_pipeline(&mut self.pipeline);
        self.gpu.destroy_command_encoder(&mut self.encoder);
    }
}

fn validate_streams(streams: impl Iterator<Item = usize>, count: usize) {
    let mut seen = vec![false; count];
    for stream in streams {
        assert!(
            stream < count && !seen[stream],
            "invalid or repeated stream"
        );
        seen[stream] = true;
    }
}

fn padded_size(bytes: u32) -> u32 {
    bytes
        .checked_next_multiple_of(4)
        .expect("frame padding overflow")
}

fn validate_shape(streams: usize, size: usize, patch: usize) {
    assert!(streams > 0 && size > 0 && patch > 0 && size.is_multiple_of(patch));
    let bytes = size
        .checked_mul(size)
        .and_then(|n| n.checked_mul(3 * size_of::<f32>()))
        .and_then(|n| n.checked_mul(streams))
        .expect("patch tensor size overflow");
    assert!(
        u32::try_from(bytes).is_ok(),
        "patch tensor exceeds GPU address range"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_checks_stride_and_address_range() {
        assert_eq!(
            FrameLayout::new(3, 2, 16, PixelFormat::Bgra8).byte_len(),
            28
        );
        for (w, h, stride) in [(0, 1, 4), (1, 0, 4), (2, 1, 4), (1, u32::MAX, 4)] {
            assert!(
                std::panic::catch_unwind(|| FrameLayout::new(w, h, stride, PixelFormat::Rgba8))
                    .is_err()
            );
        }
    }

    #[test]
    fn arrivals_must_be_distinct_and_in_range() {
        validate_streams([2, 0].into_iter(), 3);
        for streams in [[0, 0], [0, 3]] {
            assert!(std::panic::catch_unwind(|| validate_streams(streams.into_iter(), 3)).is_err());
        }
    }

    #[test]
    fn output_shape_and_padding_cannot_overflow() {
        validate_shape(6, 224, 16);
        assert_eq!(padded_size(17), 20);
        for shape in [
            (0, 224, 16),
            (1, 224, 0),
            (1, 223, 16),
            (usize::MAX, 224, 16),
            (1, 65536, 16),
        ] {
            assert!(
                std::panic::catch_unwind(|| validate_shape(shape.0, shape.1, shape.2)).is_err()
            );
        }
        assert!(std::panic::catch_unwind(|| padded_size(u32::MAX)).is_err());
    }

    fn check_device(session: &Session) {
        let device = session.device_information();
        eprintln!("pixel preprocessing device: {device:?}");
        if let Ok(driver) = std::env::var("KINDLE_GPU_DRIVER") {
            assert_eq!(std::env::var("MEGANEURA_DEVICE_ID").unwrap(), "0x2c02");
            assert_eq!(device.device_name, "NVIDIA GeForce RTX 5080");
            assert_eq!(device.driver_name, "NVIDIA");
            assert_eq!(device.driver_info, driver);
            assert!(!device.is_software_emulated);
            let memory = session.device_memory_stats().unwrap();
            assert!(
                memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 * 1024 * 1024 * 1024
            );
            eprintln!(
                "pixel memory: usage={} budget={}",
                memory.usage_bytes, memory.budget_bytes
            );
        }
    }

    fn reference(rgb: &[u8], width: usize, height: usize, size: usize, patch: usize) -> Vec<f32> {
        use crate::vision::preprocess::{
            IMAGE_MEAN, IMAGE_STD, letterbox_geometry, patches_from_pixels_chw,
        };
        let [scaled_width, scaled_height, offset_x, offset_y] =
            letterbox_geometry(width, height, size);
        let mut chw = vec![0.0; 3 * size * size];
        // Independent F64 reference; no intermediate 8-bit rounding.
        for y in 0..scaled_height {
            let sy = ((y as f64 + 0.5) * height as f64 / scaled_height as f64 - 0.5)
                .clamp(0.0, (height - 1) as f64);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(height - 1);
            let my = sy - y0 as f64;
            for x in 0..scaled_width {
                let sx = ((x as f64 + 0.5) * width as f64 / scaled_width as f64 - 0.5)
                    .clamp(0.0, (width - 1) as f64);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let mx = sx - x0 as f64;
                for c in 0..3 {
                    let sample = |x, y| f64::from(rgb[(y * width + x) * 3 + c]);
                    let value = sample(x0, y0) * (1.0 - mx) * (1.0 - my)
                        + sample(x1, y0) * mx * (1.0 - my)
                        + sample(x0, y1) * (1.0 - mx) * my
                        + sample(x1, y1) * mx * my;
                    chw[c * size * size + (y + offset_y) * size + x + offset_x] =
                        ((value / 255.0 - f64::from(IMAGE_MEAN[c])) / f64::from(IMAGE_STD[c]))
                            as f32;
                }
            }
        }
        patches_from_pixels_chw(&chw, size, patch)
    }

    fn check_patches(session: &mut Session, expected: &[f32], label: &str) -> Vec<f32> {
        session.step();
        session.wait();
        let actual = session.read_output(expected.len());
        let mut changed = 0;
        let mut worst = 0.0_f32;
        for (&negative, &value) in actual.iter().zip(expected) {
            let error = (-negative - value).abs();
            assert!(error.is_finite());
            worst = worst.max(error);
            changed += usize::from(error > 1e-5);
        }
        // Less than .03 RGB8 levels, including coordinate arithmetic at640x480.
        assert!(worst < 5e-4, "{label}: max error {worst}");
        eprintln!(
            "pixels {label}: values={} changed={changed} max_abs={worst}",
            expected.len()
        );
        check_device(session);
        actual
    }

    #[test]
    #[ignore = "requires GPU; also exercised on lavapipe in CI"]
    fn gpu_rgb64_resize_matches_scalar_reference() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = meganeura::Graph::new();
        let input = graph.input("observation", &[2, 12288]);
        let output = graph.neg(input);
        graph.set_outputs(vec![output]);
        let mut session = meganeura::build(
            &graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                gpu: Some(Arc::clone(&gpu)),
                ..Default::default()
            },
        )
        .0;
        let mut pixels = GpuPreprocessor::rgb64(Arc::clone(&gpu), 2);
        for (width, height) in [
            (160, 210),
            (64, 64),
            (13, 7),
            (1, 19),
            (640, 480),
            (641, 479),
            (1, 1),
        ] {
            let rgb = (0..width * height * 3)
                .map(|i| ((i * 37 + i / 43) % 256) as u8)
                .collect::<Vec<_>>();
            let mut expected = vec![0.0; 2 * 12288];
            let resized = rgb_filter::reference(&rgb, width, height);
            for y in 0..64 {
                for x in 0..64 {
                    for c in 0..3 {
                        expected[12288 + c * 4096 + y * 64 + x] =
                            f32::from(resized[(y * 64 + x) * 3 + c]) / 255.0 - 0.5;
                    }
                }
            }
            pixels.cpu_frames(&mut session, &[(1, &rgb, width, height)]);
            let uploaded = check_patches(&mut session, &expected, "RGB64 CPU source");
            let buffer = gpu.create_buffer(gpu::BufferDesc {
                name: "rgb64_test_source",
                size: rgb.len().next_multiple_of(4) as u64,
                memory: gpu::Memory::Shared,
            });
            unsafe {
                std::ptr::write_bytes(buffer.data(), 0, buffer.size() as usize);
                std::ptr::copy_nonoverlapping(rgb.as_ptr(), buffer.data(), rgb.len());
            }
            let frame = unsafe {
                GpuFrame::from_buffer(
                    &gpu,
                    &buffer,
                    0,
                    FrameLayout::new(
                        width as u32,
                        height as u32,
                        (width * 3) as u32,
                        PixelFormat::Rgb8,
                    ),
                )
            };
            pixels.gpu_frames(&mut session, &[(1, frame)]);
            let resident = check_patches(&mut session, &expected, "RGB64 GPU source");
            assert_eq!(resident, uploaded);
            pixels.wait();
            gpu.destroy_buffer(buffer);
        }
    }

    #[test]
    #[ignore = "requires GPU; also exercised on lavapipe in CI"]
    fn gpu_pixels_match_cpu_reference_and_resident_buffers() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        const SIZE: usize = 224;
        const PATCH: usize = 16;
        const LEN: usize = 3 * SIZE * SIZE;
        let mut graph = meganeura::Graph::new();
        let input = graph.input("patches", &[3, LEN]);
        let negative = graph.neg(input);
        graph.set_outputs(vec![negative]);
        let mut session = meganeura::build(
            &graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                gpu: Some(Arc::clone(&gpu)),
                ..Default::default()
            },
        )
        .0;
        let mut pixels = GpuPreprocessor::new(Arc::clone(&gpu), 3, SIZE, PATCH);
        check_device(&session);
        let mut expected = vec![0.0; 3 * LEN];
        for (width, height) in [
            (160, 210),
            (64, 64),
            (224, 224),
            (80, 64),
            (13, 7),
            (7, 13),
            (1, 1),
            (1, 19),
            (19, 1),
            (640, 480),
        ] {
            let rgb: Vec<u8> = (0..width * height * 3)
                .map(|i| ((i * 37 + i / 43) % 256) as u8)
                .collect();
            let reference = reference(&rgb, width, height, SIZE, PATCH);
            for stream in [2, 0, 1] {
                expected[stream * LEN..(stream + 1) * LEN].copy_from_slice(&reference);
            }
            pixels.cpu_frames(
                &mut session,
                &[
                    (2, &rgb, width, height),
                    (0, &rgb, width, height),
                    (1, &rgb, width, height),
                ],
            );
            let uploaded = check_patches(
                &mut session,
                &expected,
                &format!("CPU upload {width}x{height}"),
            );

            for format in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Bgra8] {
                let stride = width as u32 * format.channels() + 7;
                let layout = FrameLayout::new(width as u32, height as u32, stride, format);
                // Odd byte offset, padded rows and a poisoned tail verify byte addressing.
                let offset = 13;
                let count = padded_size(offset + layout.byte_len()) as usize + 16;
                let mut packed = vec![0xcd; count];
                for y in 0..height {
                    for x in 0..width {
                        let start =
                            offset as usize + y * stride as usize + x * format.channels() as usize;
                        for c in 0..3 {
                            let component = if format == PixelFormat::Bgra8 {
                                2 - c
                            } else {
                                c
                            };
                            packed[start + component] = rgb[(y * width + x) * 3 + c];
                        }
                    }
                }
                let upload = gpu.create_buffer(gpu::BufferDesc {
                    name: "test_upload",
                    size: count as u64,
                    memory: gpu::Memory::Shared,
                });
                let resident = gpu.create_buffer(gpu::BufferDesc {
                    name: "test_resident",
                    size: count as u64,
                    memory: gpu::Memory::Device,
                });
                unsafe {
                    std::ptr::copy_nonoverlapping(packed.as_ptr(), upload.data(), count);
                }
                let mut encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
                    name: "test_capture",
                    buffer_count: 1,
                    manual_barriers: false,
                });
                encoder.start();
                encoder.transfer("capture").copy_buffer_to_buffer(
                    upload.into(),
                    resident.into(),
                    count as u64,
                );
                let capture = gpu.submit(&mut encoder);
                let frame =
                    unsafe { GpuFrame::from_buffer(&gpu, &resident, u64::from(offset), layout) };
                // Consumer follows the producer on the same queue, without a host wait.
                pixels.gpu_frames(&mut session, &[(2, frame)]);
                expected[..2 * LEN].fill(0.0);
                let actual = check_patches(
                    &mut session,
                    &expected,
                    &format!("resident {format:?} {width}x{height}"),
                );
                assert_eq!(&actual[..2 * LEN], &[0.0; 2 * LEN]);
                assert_eq!(&actual[2 * LEN..], &uploaded[2 * LEN..]);
                pixels.wait();
                assert!(gpu.wait_for(&capture, !0).unwrap());
                gpu.destroy_command_encoder(&mut encoder);
                gpu.destroy_buffer(resident);
                gpu.destroy_buffer(upload);
            }
        }
    }
}
