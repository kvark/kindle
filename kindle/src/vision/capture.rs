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

const METADATA: u32 = 0x32505347;
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
    allocation: gpu::ExternalMemoryAllocation,
}

impl CaptureInfo {
    fn parse(bytes: &[u8; 88]) -> io::Result<Self> {
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
            buffer_size: u64_at(48),
            allocation: gpu::ExternalMemoryAllocation {
                size: u64_at(28),
                offset: 0,
                memory_type_index: u32_at(44),
                device_uuid: bytes[56..72].try_into().unwrap(),
                driver_uuid: bytes[72..88].try_into().unwrap(),
            },
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
            || info.buffer_size > info.allocation.size
            || info.allocation.memory_type_index >= 32
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
    /// The trusted local producer must implement Dullahan GPU_SYNC: truthful
    /// allocation metadata, release-to-EXTERNAL plus a completed producer fence
    /// before FRAME, and no reuse until ACK. Legacy SHM-ready captures are refused.
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
            let mut packet = [0u8; 88];
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
                        memory: gpu::Memory::External(gpu::ExternalMemorySource::Fd(Some((
                            handles[0].as_raw_fd(),
                            info.allocation,
                        )))),
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
                    unsafe {
                        self.gpu.acquire_external_buffer(
                            &mut self.encoder,
                            buffer.at(offset),
                            info.stride,
                        );
                    }
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
        let Some((mut packet, offset, acquired)) = self.pending.take() else {
            return Ok(());
        };
        assert!(
            self.gpu
                .wait_for(&acquired, !0)
                .expect("capture acquire wait failed")
        );
        let (buffer, info) = self.imported.unwrap();
        self.encoder.start();
        unsafe {
            self.gpu
                .release_external_buffer(&mut self.encoder, buffer.at(offset), info.stride);
        }
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
            let Some(gpu::ExternalMemorySource::Fd(Some((fd, info)))) =
                gpu.get_external_buffer_source(ring)
            else {
                panic!("missing export FD")
            };
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            assert_eq!(info.offset, 0);
            let mut encoder = gpu.create_command_encoder(gpu::CommandEncoderDesc {
                name: "test_producer",
                buffer_count: 1,
                manual_barriers: false,
            });
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(30)))
                .unwrap();
            let mut metadata = [0u8; 88];
            for (offset, value) in [
                (0, METADATA),
                (4, 32),
                (8, 16),
                (12, 37),
                (16, 3),
                (44, info.memory_type_index),
            ] {
                metadata[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            for (offset, value) in [(20, 2048), (28, info.size), (36, 64), (48, 64 + 3 * 2048)] {
                metadata[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
            metadata[56..72].copy_from_slice(&info.device_uuid);
            metadata[72..88].copy_from_slice(&info.driver_uuid);
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
                if generation > 3 {
                    unsafe {
                        gpu.acquire_external_buffer(&mut encoder, range, 2048);
                    }
                }
                encoder
                    .transfer("write_pattern")
                    .copy_buffer_to_buffer(upload.into(), range, 2048);
                unsafe {
                    gpu.release_external_buffer(&mut encoder, range, 2048);
                }
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

    fn header() -> [u8; 88] {
        let mut data = [0; 88];
        for (offset, value) in [(0, METADATA), (4, 17), (8, 11), (12, 44), (16, 3), (44, 2)] {
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (offset, value) in [(20, 17u64 * 11 * 4), (28, 4096), (36, 64), (48, 4096)] {
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
        for offset in [0, 4, 8, 12, 16, 20, 28, 36, 44, 48] {
            let mut data = header();
            let value = if offset == 28 { 0 } else { u32::MAX };
            data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(CaptureInfo::parse(&data).is_err(), "offset {offset}");
        }
    }
}
