//! Dullahan's opt-in fenced Vulkan capture protocol. Pixels never visit the CPU.
//! A frame is released and acknowledged only after this queue has finished using
//! it. The producer blocks reuse until that acknowledgement arrives.

use std::{
    io::{self, IoSliceMut, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
        unix::net::UnixStream,
    },
    path::Path,
    sync::Arc,
    time::Duration,
};

use blade_graphics as gpu;
use nix::sys::socket::{ControlMessageOwned, MsgFlags, recvmsg};

use super::preprocess_gpu::{FrameLayout, GpuFrame, PixelFormat};

const METADATA: u32 = 0x34505347;
const FRAME: u32 = 0x32465247;
const STOP: u32 = 0x32544f53;

#[derive(Clone, Copy, Debug)]
pub struct CaptureInfo {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub stride: u64,
    pub data_offset: u64,
    pub format: PixelFormat,
    buffer_size: u64,
}

impl CaptureInfo {
    fn parse(bytes: &[u8; 44]) -> io::Result<Self> {
        let u32_at = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let u64_at = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        let format = match u32_at(12) {
            37 | 43 => PixelFormat::Rgba8,
            44 | 50 => PixelFormat::Bgra8,
            _ => return Err(io::Error::other("capture must be RGBA8 or BGRA8")),
        };
        let info = Self {
            width: u32_at(4),
            height: u32_at(8),
            frames: u32_at(16),
            stride: u64_at(20),
            data_offset: u64_at(36),
            format,
            buffer_size: u64_at(28),
        };
        let frame_bytes = u64::from(info.width)
            .checked_mul(u64::from(info.height))
            .and_then(|n| n.checked_mul(4));
        let end = u64::from(info.frames)
            .checked_mul(info.stride)
            .and_then(|n| n.checked_add(info.data_offset));
        if u32_at(0) != METADATA
            || info.width == 0
            || info.height == 0
            || info.frames == 0
            || info.width > u32::MAX / 4
            || !info.data_offset.is_multiple_of(4)
            || !info.stride.is_multiple_of(4)
            || frame_bytes.is_none_or(|n| n != info.stride)
            || end.is_none_or(|n| n > info.buffer_size || n > u64::from(u32::MAX))
            || info.buffer_size > u64::from(u32::MAX)
        {
            return Err(io::Error::other("invalid capture allocation geometry"));
        }
        Ok(info)
    }
}

pub struct CaptureStream {
    gpu: Arc<gpu::Context>,
    stream: UnixStream,
    encoder: gpu::CommandEncoder,
    imported: Option<(gpu::Buffer, CaptureInfo)>,
    generation: u64,
    pending: Option<([u8; 16], u64, gpu::SyncPoint)>,
}

impl CaptureStream {
    /// # Safety
    /// The trusted local producer must implement Dullahan GPU_SYNC v4 on the
    /// same physical device and driver, using Blade's matching external-buffer
    /// allocation recipe. It must release the whole buffer to EXTERNAL and
    /// complete its fence before FRAME, and prevent reuse until ACK. Older
    /// protocols are refused.
    pub unsafe fn connect(
        context: Arc<gpu::Context>,
        path: impl AsRef<Path>,
        timeout: Duration,
    ) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "kindle_capture_handoff",
            buffer_count: 1,
            manual_barriers: false,
        });
        Ok(Self {
            gpu: context,
            stream,
            encoder,
            imported: None,
            generation: 0,
            pending: None,
        })
    }

    pub fn next_frame(&mut self) -> io::Result<CapturedFrame<'_>> {
        if self.pending.is_some() {
            return Err(io::Error::other("previous capture lease was not finished"));
        }
        loop {
            // Read just the packet tag with recvmsg so metadata FDs cannot be
            // consumed unnoticed by a buffered read. Unix streams may split it.
            let mut packet = [0u8; 44];
            let mut iov = [IoSliceMut::new(&mut packet[..4])];
            let mut ancillary = nix::cmsg_space!([RawFd; 4]);
            let message = recvmsg::<()>(
                self.stream.as_raw_fd(),
                &mut iov,
                Some(&mut ancillary),
                MsgFlags::MSG_CMSG_CLOEXEC,
            )
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
            let read = message.bytes;
            let truncated = message.flags.contains(MsgFlags::MSG_CTRUNC);
            let mut handles = Vec::new();
            for control in message.cmsgs() {
                if let ControlMessageOwned::ScmRights(fds) = control {
                    handles.extend(
                        fds.into_iter()
                            .map(|fd| unsafe { OwnedFd::from_raw_fd(fd) }),
                    );
                }
            }
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "capture disconnected",
                ));
            }
            if truncated {
                return Err(io::Error::other("truncated capture descriptors"));
            }
            self.stream.read_exact(&mut packet[read..4])?;
            match u32::from_le_bytes(packet[..4].try_into().unwrap()) {
                METADATA => {
                    if handles.len() != 1 {
                        return Err(io::Error::other("capture needs exactly one allocation FD"));
                    }
                    self.stream.read_exact(&mut packet[4..])?;
                    let info = CaptureInfo::parse(&packet)?;
                    let buffer = self.gpu.create_buffer(gpu::BufferDesc {
                        name: "kindle_capture",
                        size: info.buffer_size,
                        memory: gpu::Memory::External(gpu::ExternalMemorySource::Fd(Some(
                            handles[0].as_raw_fd(),
                        ))),
                    });
                    if let Some((old, _)) = self.imported.replace((buffer, info)) {
                        self.gpu.destroy_buffer(old);
                    }
                }
                FRAME => {
                    if !handles.is_empty() {
                        return Err(io::Error::other("unexpected frame FD"));
                    }
                    self.stream.read_exact(&mut packet[4..16])?;
                    let slot = u32::from_le_bytes(packet[4..8].try_into().unwrap());
                    let generation = u64::from_le_bytes(packet[8..16].try_into().unwrap());
                    let (buffer, info) = self
                        .imported
                        .ok_or_else(|| io::Error::other("frame precedes capture metadata"))?;
                    if slot >= info.frames || generation <= self.generation {
                        return Err(io::Error::other("stale or out-of-range capture frame"));
                    }
                    self.generation = generation;
                    let offset = info.data_offset + u64::from(slot) * info.stride;
                    self.encoder.start();
                    self.encoder.acquire_external_buffer(buffer);
                    let acquired = self.gpu.submit(&mut self.encoder);
                    self.pending = Some((packet[..16].try_into().unwrap(), offset, acquired));
                    return Ok(CapturedFrame { capture: self });
                }
                _ => {
                    return Err(io::Error::other(
                        "capture requires the fenced GPU_SYNC protocol, not legacy SHM synchronization",
                    ));
                }
            }
        }
    }

    fn release(&mut self, stop: bool) -> io::Result<()> {
        let Some((mut packet, _, acquired)) = self.pending.take() else {
            return Ok(());
        };
        assert!(
            self.gpu
                .wait_for(&acquired, !0)
                .expect("capture acquire wait failed")
        );
        let (buffer, _) = self.imported.unwrap();
        self.encoder.start();
        self.encoder.release_external_buffer(buffer);
        let released = self.gpu.submit(&mut self.encoder);
        assert!(
            self.gpu
                .wait_for(&released, !0)
                .expect("capture release wait failed")
        );
        if stop {
            packet[..4].copy_from_slice(&STOP.to_le_bytes());
        }
        self.stream.write_all(&packet)
    }
}

impl Drop for CaptureStream {
    fn drop(&mut self) {
        let _ = self.release(true);
        self.gpu.destroy_command_encoder(&mut self.encoder);
        if let Some((buffer, _)) = self.imported.take() {
            self.gpu.destroy_buffer(buffer);
        }
    }
}

/// Exclusive producer lease. Dropping without finish stops the stream after
/// releasing GPU ownership, including on a policy panic; it never frees a slot early.
pub struct CapturedFrame<'a> {
    capture: &'a mut CaptureStream,
}

impl CapturedFrame<'_> {
    pub fn info(&self) -> CaptureInfo {
        self.capture.imported.unwrap().1
    }

    pub fn frame(&self) -> GpuFrame<'_> {
        let (buffer, info) = self.capture.imported.as_ref().unwrap();
        let layout = FrameLayout::new(info.width, info.height, info.width * 4, info.format);
        let offset = self.capture.pending.as_ref().unwrap().1;
        unsafe { GpuFrame::from_buffer(&self.capture.gpu, buffer, offset, layout) }
    }

    /// Explicit full-frame diagnostic readback, never used by the acting path.
    pub fn read_rgb8(&self) -> crate::RgbFrame {
        let (buffer, info) = self.capture.imported.unwrap();
        let offset = self.capture.pending.as_ref().unwrap().1;
        let gpu = &self.capture.gpu;
        let target = gpu.create_buffer(gpu::BufferDesc {
            name: "capture_diagnostic",
            size: info.stride,
            memory: gpu::Memory::Shared,
        });
        let mut encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
            name: "capture_diagnostic",
            buffer_count: 1,
            manual_barriers: false,
        });
        encoder.start();
        encoder
            .transfer("capture_diagnostic")
            .copy_buffer_to_buffer(buffer.at(offset), target.into(), info.stride);
        assert!(
            gpu.wait_for(&gpu.submit(&mut encoder), !0)
                .expect("diagnostic readback failed")
        );
        let rgba = unsafe { std::slice::from_raw_parts(target.data(), info.stride as usize) };
        let rgb = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| match info.format {
                PixelFormat::Rgba8 => [pixel[0], pixel[1], pixel[2]],
                PixelFormat::Bgra8 => [pixel[2], pixel[1], pixel[0]],
                PixelFormat::Rgb8 => unreachable!("capture format checked at import"),
            })
            .collect();
        gpu.destroy_command_encoder(&mut encoder);
        gpu.destroy_buffer(target);
        crate::RgbFrame::new(info.width as usize, info.height as usize, rgb)
    }

    /// Acknowledge only after all same-queue readers have completed. Set stop on
    /// the last frame so a normal shutdown is distinct from a broken consumer.
    pub fn finish(self, stop: bool) -> io::Result<()> {
        self.capture.release(stop)
    }
}

impl Drop for CapturedFrame<'_> {
    fn drop(&mut self) {
        let _ = self.capture.release(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires native Vulkan, Xvfb, vkcube and a built Dullahan GPU_SYNC v4 layer"]
    fn dullahan_producer_imports_matching_allocation() {
        use nix::libc;
        use std::{
            fs::{self, File},
            io::{BufRead, BufReader},
            os::unix::process::CommandExt,
            process::{Child, Command, Stdio},
            thread,
            time::Instant,
        };

        struct OwnedChild(Child);
        impl OwnedChild {
            fn spawn(command: &mut Command) -> Self {
                let parent = std::process::id();
                unsafe {
                    command.pre_exec(move || {
                        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                            return Err(io::Error::last_os_error());
                        }
                        if libc::getppid() as u32 != parent {
                            return Err(io::Error::other("parent exited"));
                        }
                        Ok(())
                    });
                }
                Self(command.spawn().unwrap())
            }
        }
        impl Drop for OwnedChild {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }

        let root = std::path::PathBuf::from(std::env::var("KINDLE_CAPTURE_TEST_OUTPUT").unwrap());
        fs::create_dir_all(&root).unwrap();
        let layer = std::path::PathBuf::from(std::env::var("KINDLE_DULLAHAN").unwrap());
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let device = gpu.device_information();
        assert!(!device.is_software_emulated);
        assert_eq!(
            device.device_name,
            std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
        );
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        let mut display = OwnedChild::spawn(
            Command::new("Xvfb")
                .args([
                    "-displayfd",
                    "1",
                    "-screen",
                    "0",
                    "160x120x24",
                    "-nolisten",
                    "tcp",
                ])
                .stdout(Stdio::piped())
                .stderr(File::create_new(root.join("xvfb.log")).unwrap()),
        );
        let mut number = String::new();
        BufReader::new(display.0.stdout.take().unwrap())
            .read_line(&mut number)
            .unwrap();
        let number: u16 = number.trim().parse().unwrap();
        let socket = root.join("capture.sock");
        let shm = format!("kindle-capture-test-{}", std::process::id());
        let mut producer = OwnedChild::spawn(
            Command::new("vkcube")
                .args(["--width", "160", "--height", "120"])
                .env("DISPLAY", format!(":{number}"))
                .env("VK_ADD_LAYER_PATH", &layer)
                .env("LD_LIBRARY_PATH", layer.join("target/release"))
                .env(
                    "VK_INSTANCE_LAYERS",
                    "VK_LAYER_PRIVATE_dullahan:VK_LAYER_KHRONOS_validation",
                )
                .env("VK_LAYER_DULLAHAN_MODE", "opaque")
                .env("VK_LAYER_DULLAHAN_SHM_NAME", &shm)
                .env("VK_LAYER_DULLAHAN_GPU_SOCKET", &socket)
                .env("VK_LAYER_DULLAHAN_GPU_SYNC", "1")
                .env("RUST_LOG", "info")
                .stdout(File::create_new(root.join("producer.stdout")).unwrap())
                .stderr(File::create_new(root.join("producer.stderr")).unwrap()),
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        while !socket.exists() {
            assert!(producer.0.try_wait().unwrap().is_none(), "producer exited");
            assert!(Instant::now() < deadline, "producer startup timed out");
            thread::sleep(Duration::from_millis(20));
        }
        let mut capture =
            unsafe { CaptureStream::connect(gpu, &socket, Duration::from_secs(30)).unwrap() };
        let mut slots = std::collections::BTreeSet::new();
        for index in 0..12 {
            let frame = capture.next_frame().unwrap();
            let info = frame.info();
            assert_eq!((info.width, info.height), (160, 120));
            let offset = frame.capture.pending.as_ref().unwrap().1;
            slots.insert((offset - info.data_offset) / info.stride);
            let pixels = frame.read_rgb8();
            assert!(
                pixels
                    .pixels()
                    .iter()
                    .any(|&value| value != pixels.pixels()[0])
            );
            frame.finish(index == 11).unwrap();
        }
        assert!(slots.len() > 1, "capture did not rotate ring slots");
        eprintln!(
            "Dullahan capture: 12 frames, {} slots, 160x120",
            slots.len()
        );
        drop(capture);
        drop(producer);
        // The owned producer is killed after STOP; clean its private IPC names.
        fs::remove_file(format!("/dev/shm/{shm}")).unwrap();
        fs::remove_file(socket).unwrap();
        for name in ["producer.stdout", "producer.stderr"] {
            let log = fs::read_to_string(root.join(name)).unwrap();
            assert!(
                !log.contains("Validation Error") && !log.contains("SYNC-HAZARD"),
                "producer validation failed; inspect {name}"
            );
        }
    }

    #[test]
    #[ignore = "requires two native Vulkan devices/queues with OPAQUE_FD support"]
    fn fenced_external_ring_roundtrips_without_stale_frames() {
        use nix::sys::socket::{ControlMessage, sendmsg};
        use std::{io::IoSlice, os::unix::net::UnixListener, thread};

        let path = std::env::temp_dir().join(format!("kindle-capture-{}.sock", std::process::id()));
        let listener = UnixListener::bind(&path).unwrap();
        let producer = thread::spawn(move || {
            let gpu = crate::init_gpu_context().unwrap();
            let ring = gpu.create_buffer(gpu::BufferDesc {
                name: "external_test_ring",
                size: 64 + 3 * 2048,
                memory: gpu::Memory::External(gpu::ExternalMemorySource::Fd(None)),
            });
            let upload = gpu.create_buffer(gpu::BufferDesc {
                name: "test_pattern",
                size: 2048,
                memory: gpu::Memory::Shared,
            });
            let Some(gpu::ExternalMemorySource::Fd(Some(fd))) =
                gpu.get_external_buffer_source(ring)
            else {
                panic!("missing export FD")
            };
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let mut encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "test_producer",
                buffer_count: 1,
                manual_barriers: false,
            });
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(30)))
                .unwrap();
            let mut metadata = [0u8; 44];
            for (offset, value) in [(0, METADATA), (4, 32), (8, 16), (12, 37), (16, 3)] {
                metadata[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            for (offset, value) in [(20, 2048u64), (28, 64 + 3 * 2048), (36, 64)] {
                metadata[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
            // Split the tag and metadata deliberately, including the SCM_RIGHTS packet.
            assert_eq!(
                sendmsg::<()>(
                    socket.as_raw_fd(),
                    &[IoSlice::new(&metadata[..2])],
                    &[ControlMessage::ScmRights(&[fd.as_raw_fd()])],
                    MsgFlags::MSG_NOSIGNAL,
                    None
                )
                .unwrap(),
                2
            );
            socket.write_all(&metadata[2..]).unwrap();
            for generation in 1..=12u64 {
                let slot = (generation - 1) % 3;
                let range = ring.at(64 + slot * 2048);
                unsafe {
                    std::slice::from_raw_parts_mut(upload.data(), 2048).fill(generation as u8);
                }
                encoder.start();
                if generation > 1 {
                    encoder.acquire_external_buffer(ring);
                }
                encoder
                    .transfer("write_pattern")
                    .copy_buffer_to_buffer(upload.into(), range, 2048);
                encoder.release_external_buffer(ring);
                assert!(gpu.wait_for(&gpu.submit(&mut encoder), !0).unwrap());
                let mut packet = [0u8; 16];
                packet[..4].copy_from_slice(&FRAME.to_le_bytes());
                packet[4..8].copy_from_slice(&(slot as u32).to_le_bytes());
                packet[8..].copy_from_slice(&generation.to_le_bytes());
                socket.write_all(&packet).unwrap();
                let mut ack = [0; 16];
                socket.read_exact(&mut ack).unwrap();
                if generation == 12 {
                    packet[..4].copy_from_slice(&STOP.to_le_bytes());
                }
                assert_eq!(ack, packet);
            }
            gpu.destroy_command_encoder(&mut encoder);
            gpu.destroy_buffer(upload);
            gpu.destroy_buffer(ring);
        });
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let device = gpu.device_information();
        assert!(!device.is_software_emulated);
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            assert_eq!(device.device_name, expected);
        }
        eprintln!("capture device={}", device.device_name);
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
        let mut stream = unsafe {
            CaptureStream::connect(Arc::clone(&gpu), &path, Duration::from_secs(30)).unwrap()
        };
        let readback = gpu.create_buffer(gpu::BufferDesc {
            name: "capture_oracle",
            size: 2048,
            memory: gpu::Memory::Shared,
        });
        let mut encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
            name: "capture_oracle",
            buffer_count: 1,
            manual_barriers: false,
        });
        for generation in 1..=12u8 {
            let frame = stream.next_frame().unwrap();
            assert_eq!((frame.info().width, frame.info().height), (32, 16));
            let (buffer, _) = frame.capture.imported.unwrap();
            assert!(gpu.get_external_buffer_source(buffer).is_none());
            let offset = frame.capture.pending.as_ref().unwrap().1;
            encoder.start();
            encoder.transfer("capture_oracle").copy_buffer_to_buffer(
                buffer.at(offset),
                readback.into(),
                2048,
            );
            assert!(gpu.wait_for(&gpu.submit(&mut encoder), !0).unwrap());
            assert!(
                unsafe { std::slice::from_raw_parts(readback.data(), 2048) }
                    .iter()
                    .all(|&v| v == generation)
            );
            if generation == 12 {
                drop(frame);
            } else {
                frame.finish(false).unwrap();
            }
        }
        producer.join().unwrap();
        drop(stream);
        gpu.destroy_command_encoder(&mut encoder);
        gpu.destroy_buffer(readback);
        std::fs::remove_file(path).unwrap();
    }

    fn header() -> [u8; 44] {
        let mut data = [0; 44];
        for (offset, value) in [(0, METADATA), (4, 17), (8, 11), (12, 44), (16, 3)] {
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (offset, value) in [(20, 17u64 * 11 * 4), (28, 4096), (36, 64)] {
            data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
        data
    }

    #[test]
    fn capture_rejects_legacy_formats_overflow_and_short_allocations() {
        assert_eq!(
            CaptureInfo::parse(&header()).unwrap().format,
            PixelFormat::Bgra8
        );
        for magic in [0x32505347u32, 0x33505347, 0x554e5347] {
            let mut data = header();
            data[..4].copy_from_slice(&magic.to_le_bytes());
            assert!(CaptureInfo::parse(&data).is_err());
        }
        for offset in [0, 4, 8, 12, 16, 20, 28, 36] {
            let mut data = header();
            let value = if offset == 28 { 0 } else { u32::MAX };
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(CaptureInfo::parse(&data).is_err(), "offset {offset}");
        }
    }
}
