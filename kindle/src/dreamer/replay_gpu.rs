//! Paged device replay. Collection copies features/state on GPU; only sampled
//! training batches cross the existing learner's host target-building boundary.

use std::sync::Arc;

use blade_graphics as gpu;
use meganeura::Session;

use super::{DreamerConfig, device_copy::DeviceCopies, readback::Readback};

const PAGE_FRAMES: usize = 256;

pub(super) struct DeviceReplay {
    gpu: Arc<gpu::Context>,
    copies: DeviceCopies,
    pub readback: Readback,
    pages: Vec<gpu::Buffer>,
    capacity: usize,
    width: usize,
    observation: usize,
    deter: usize,
}

impl DeviceReplay {
    pub fn new(gpu: Arc<gpu::Context>, config: &DreamerConfig) -> Self {
        Self {
            copies: DeviceCopies::new(Arc::clone(&gpu)),
            readback: Readback::new(Arc::clone(&gpu)),
            gpu,
            pages: Vec::new(),
            capacity: config.replay_capacity,
            width: config.observation_dim() + config.feature_dim(),
            observation: config.observation_dim(),
            deter: config.network().deter,
        }
    }

    pub fn store(&mut self, session: &Session, slots: &[(usize, usize)]) {
        if slots.is_empty() {
            return;
        }
        let last = slots.iter().map(|&(_, slot)| slot).max().unwrap();
        assert!(last < self.capacity);
        while self.pages.len() <= last / PAGE_FRAMES {
            let rows = PAGE_FRAMES.min(self.capacity - self.pages.len() * PAGE_FRAMES);
            self.pages.push(self.gpu.create_buffer(gpu::BufferDesc {
                name: "kindle_replay",
                size: (rows * self.width * 4) as u64,
                memory: gpu::Memory::Device,
            }));
        }
        let mut copies = Vec::with_capacity(slots.len() * 3);
        for &(stream, slot) in slots {
            for (name, offset, width) in [
                ("observation", 0, self.observation),
                ("previous_deter", self.observation, self.deter),
                (
                    "previous_stoch",
                    self.observation + self.deter,
                    self.width - self.observation - self.deter,
                ),
            ] {
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
    }
}

impl Drop for DeviceReplay {
    fn drop(&mut self) {
        self.copies.wait();
        for page in self.pages.drain(..) {
            self.gpu.destroy_buffer(page);
        }
    }
}
