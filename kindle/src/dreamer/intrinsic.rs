//! A bounded novelty control over the frozen visual representation.
//!
//! Fixed random-hyperplane hashing groups nearby observations. Collisions
//! conservatively reduce novelty; this is visitation, not model uncertainty.

use crate::vision::{DinoObservation, fixed_projection};

const HASH_BITS: usize = 16;
const HASH_SEED: u64 = 0x6b69_6e64_6c65_7631;
pub(super) const VERSION: u32 = 2;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(super) struct VisitationState {
    version: u32,
    counts: Box<[u32]>,
    /// Fixed first observation removes common background without drifting the
    /// coordinate system as counts accumulate.
    reference: Option<Box<[f32]>>,
}

impl VisitationState {
    pub fn is_valid(&self) -> bool {
        self.version == VERSION
            && self.counts.len() == 1 << HASH_BITS
            && match &self.reference {
                Some(values) => {
                    values.len() == DinoObservation::LEN
                        && values.iter().all(|value| value.is_finite())
                }
                None => self.counts.iter().all(|count| *count == 0),
            }
    }
}

pub(super) struct VisitationBonus {
    projection: Vec<f32>,
    counts: Box<[u32]>,
    reference: Option<Box<[f32]>>,
}

impl VisitationBonus {
    pub fn new() -> Self {
        Self {
            projection: fixed_projection(DinoObservation::LEN, HASH_BITS, HASH_SEED),
            counts: vec![0; 1 << HASH_BITS].into_boxed_slice(),
            reference: None,
        }
    }

    pub fn state(&self) -> VisitationState {
        VisitationState {
            version: VERSION,
            counts: self.counts.clone(),
            reference: self.reference.clone(),
        }
    }

    pub fn restore(&mut self, state: VisitationState) {
        assert!(state.is_valid(), "incompatible visitation state");
        self.counts = state.counts;
        self.reference = state.reference;
    }

    /// Score this arrival using the previous count, then count the visit.
    /// Counts saturate; memory and the bonus remain bounded for a long life.
    pub fn observe(&mut self, observation: &DinoObservation) -> f32 {
        let mut projected = [0.0; HASH_BITS];
        let reference = self
            .reference
            .get_or_insert_with(|| observation.as_slice().into());
        for ((&value, &reference), weights) in observation
            .as_slice()
            .iter()
            .zip(reference.iter())
            .zip(self.projection.as_chunks::<HASH_BITS>().0)
        {
            for (sum, &weight) in projected.iter_mut().zip(weights) {
                *sum += (value - reference) * weight;
            }
        }
        let bucket = projected.iter().enumerate().fold(0, |bits, (bit, &value)| {
            bits | (usize::from(value >= 0.0) << bit)
        });
        let count = &mut self.counts[bucket];
        let bonus = 1.0 / (f64::from(*count) + 1.0).sqrt();
        *count = count.saturating_add(1);
        bonus as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_visits_decay_and_survive_serialization() {
        let observation = DinoObservation::from_vec(vec![0.2; DinoObservation::LEN]);
        let mut bonus = VisitationBonus::new();
        assert_eq!(bonus.observe(&observation), 1.0);
        assert!((bonus.observe(&observation) - 1.0 / 2.0_f32.sqrt()).abs() < 1e-7);
        let encoded = serde_json::to_vec(&bonus.state()).unwrap();
        let mut restored = VisitationBonus::new();
        restored.restore(serde_json::from_slice(&encoded).unwrap());
        assert_eq!(restored.observe(&observation), bonus.observe(&observation));
        assert_eq!(
            restored.observe(&DinoObservation::from_vec(vec![-0.2; DinoObservation::LEN])),
            1.0
        );
    }

    #[test]
    fn saturated_counts_never_wrap_or_create_fresh_novelty() {
        let mut bonus = VisitationBonus::new();
        bonus.counts.fill(u32::MAX);
        let observation = DinoObservation::from_vec(vec![0.0; DinoObservation::LEN]);
        let first = bonus.observe(&observation);
        assert!(first > 0.0 && first < 0.00002);
        assert_eq!(bonus.observe(&observation), first);
        assert!(bonus.counts.iter().all(|&count| count == u32::MAX));
    }

    #[test]
    fn checkpoint_rejects_changed_hash_scheme_or_size() {
        let mut state = VisitationBonus::new().state();
        assert!(state.is_valid());
        state.version += 1;
        assert!(!state.is_valid());
        state.version = VERSION;
        state.counts = Box::new([]);
        assert!(!state.is_valid());
    }

    #[test]
    fn checkpoint_requires_a_finite_fixed_reference_after_visiting() {
        let mut state = VisitationBonus::new().state();
        state.counts[0] = 1;
        assert!(!state.is_valid());
        state.reference = Some(vec![0.0; DinoObservation::LEN].into_boxed_slice());
        assert!(state.is_valid());
        state.reference.as_mut().unwrap()[0] = f32::NAN;
        assert!(!state.is_valid());
    }
}
