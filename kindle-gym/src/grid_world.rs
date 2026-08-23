use kindle::{Environment, Reward, RgbFrame, Transition};

pub const WIDTH: usize = 5;
pub const HEIGHT: usize = 5;
pub const ACTION_COUNT: usize = 4;
pub const FRAME_WIDTH: usize = 160;
pub const FRAME_HEIGHT: usize = 176;
const CELL: usize = 32;
const STATUS_HEIGHT: usize = FRAME_HEIGHT - HEIGHT * CELL;
const EPISODE_LIMIT: usize = 200;

/// A deterministic visual task with sparse food reward and an energy-based
/// terminal. Its purpose is integration testing, not benchmarking Dreamer.
pub struct GridWorld {
    position: (usize, usize),
    energy: f32,
    food_index: usize,
    steps: usize,
}

impl Default for GridWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl GridWorld {
    const FOOD: [(usize, usize); 3] = [(1, 3), (3, 1), (4, 4)];

    pub fn new() -> Self {
        Self {
            position: (0, 0),
            energy: 1.0,
            food_index: 0,
            steps: 0,
        }
    }

    pub fn position(&self) -> (usize, usize) {
        self.position
    }

    pub fn energy(&self) -> f32 {
        self.energy
    }

    pub fn render(&self) -> RgbFrame {
        let mut pixels = vec![18_u8; FRAME_WIDTH * FRAME_HEIGHT * 3];
        let energy_width = (self.energy * (FRAME_WIDTH - 8) as f32).round() as usize;
        fill_rect(
            &mut pixels,
            4,
            4,
            energy_width,
            STATUS_HEIGHT - 8,
            [48, 196, 96],
        );

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let color = if (x + y) % 2 == 0 {
                    [38, 42, 54]
                } else {
                    [45, 49, 62]
                };
                fill_rect(
                    &mut pixels,
                    x * CELL + 1,
                    STATUS_HEIGHT + y * CELL + 1,
                    CELL - 2,
                    CELL - 2,
                    color,
                );
            }
        }

        let (food_x, food_y) = Self::FOOD[self.food_index];
        fill_rect(
            &mut pixels,
            food_x * CELL + 9,
            STATUS_HEIGHT + food_y * CELL + 9,
            14,
            14,
            [244, 162, 54],
        );
        fill_rect(
            &mut pixels,
            self.position.0 * CELL + 6,
            STATUS_HEIGHT + self.position.1 * CELL + 6,
            20,
            20,
            [65, 182, 230],
        );
        RgbFrame::new(FRAME_WIDTH, FRAME_HEIGHT, pixels)
    }
}

impl Environment for GridWorld {
    fn action_count(&self) -> usize {
        ACTION_COUNT
    }

    fn reset(&mut self) -> RgbFrame {
        self.position = (0, 0);
        self.energy = 1.0;
        self.food_index = 0;
        self.steps = 0;
        self.render()
    }

    fn action_mask(&self) -> Option<Vec<bool>> {
        Some(vec![
            self.position.1 > 0,
            self.position.1 + 1 < HEIGHT,
            self.position.0 > 0,
            self.position.0 + 1 < WIDTH,
        ])
    }

    fn step(&mut self, action: usize) -> Transition {
        assert!(action < ACTION_COUNT);
        match action {
            0 if self.position.1 > 0 => self.position.1 -= 1,
            1 if self.position.1 + 1 < HEIGHT => self.position.1 += 1,
            2 if self.position.0 > 0 => self.position.0 -= 1,
            3 if self.position.0 + 1 < WIDTH => self.position.0 += 1,
            _ => {}
        }
        self.steps += 1;
        self.energy = (self.energy - 0.01).max(0.0);
        let found_food = self.position == Self::FOOD[self.food_index];
        if found_food {
            self.energy = (self.energy + 0.35).min(1.0);
            self.food_index = (self.food_index + 1) % Self::FOOD.len();
        }
        Transition {
            frame: self.render(),
            reward: Reward {
                extrinsic: if found_food { 1.0 } else { 0.0 },
                intrinsic: 0.0,
            },
            terminated: self.energy == 0.0,
            truncated: self.steps >= EPISODE_LIMIT,
        }
    }
}

fn fill_rect(pixels: &mut [u8], x: usize, y: usize, width: usize, height: usize, color: [u8; 3]) {
    assert!(x + width <= FRAME_WIDTH && y + height <= FRAME_HEIGHT);
    for row in y..y + height {
        for column in x..x + width {
            let offset = (row * FRAME_WIDTH + column) * 3;
            pixels[offset..offset + 3].copy_from_slice(&color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_and_action_mask_follow_state() {
        let mut world = GridWorld::new();
        let frame = world.reset();
        assert_eq!((frame.width(), frame.height()), (160, 176));
        assert_eq!(world.action_mask().unwrap(), vec![false, true, false, true]);
        world.step(3);
        assert_eq!(world.position(), (1, 0));
        assert_eq!(world.action_mask().unwrap(), vec![false, true, true, true]);
    }

    #[test]
    fn food_is_sparse_extrinsic_reward() {
        let mut world = GridWorld::new();
        world.reset();
        for action in [3, 1, 1] {
            assert_eq!(world.step(action).reward.extrinsic, 0.0);
        }
        assert_eq!(world.step(1).reward.extrinsic, 1.0);
    }
}
