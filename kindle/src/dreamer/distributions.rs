//! CPU-side categorical sampling and scalar targets matching DreamerV3.

use rand::Rng;

pub fn symexp(value: f32) -> f32 {
    value.signum() * value.abs().exp_m1()
}

/// DreamerV3's exponentially spaced two-hot scalar support.
#[derive(Clone, Debug)]
pub struct TwoHotBins {
    values: Box<[f32]>,
}

impl TwoHotBins {
    pub fn new(count: usize) -> Self {
        assert!(count >= 3 && count % 2 == 1);
        let half = (count - 1) / 2;
        // Construct one half and mirror it, as upstream does, so the support
        // is bit-for-bit symmetric around zero instead of merely close after
        // independent floating-point exponentiation.
        let mut values = (0..=half)
            .map(|index| symexp(-20.0 + 20.0 * index as f32 / half as f32))
            .collect::<Vec<_>>();
        let positive = (0..half)
            .rev()
            .map(|index| -values[index])
            .collect::<Vec<_>>();
        values.extend(positive);
        let values = values.into_boxed_slice();
        assert_eq!(values[half], 0.0);
        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[cfg(test)]
    fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn encode(&self, target: f32, output: &mut [f32]) {
        assert_eq!(output.len(), self.len());
        output.fill(0.0);
        let target = target.clamp(self.values[0], self.values[self.len() - 1]);
        let upper = self.values.partition_point(|value| *value < target);
        if upper == 0 {
            output[0] = 1.0;
            return;
        }
        if upper == self.len() {
            output[self.len() - 1] = 1.0;
            return;
        }
        let lower = upper - 1;
        if self.values[upper] == target {
            output[upper] = 1.0;
            return;
        }
        let low_distance = (target - self.values[lower]).abs();
        let high_distance = (self.values[upper] - target).abs();
        let total = low_distance + high_distance;
        output[lower] = high_distance / total;
        output[upper] = low_distance / total;
    }

    pub fn decode_logits(&self, logits: &[f32]) -> f32 {
        assert_eq!(logits.len(), self.len());
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials = logits
            .iter()
            .map(|logit| ((*logit - maximum) as f64).exp())
            .collect::<Vec<_>>();
        let normalizer = exponentials.iter().sum::<f64>();
        let middle = self.len() / 2;
        let mut value = exponentials[middle] / normalizer * self.values[middle] as f64;
        for lower in 0..middle {
            let upper = self.len() - 1 - lower;
            value += (exponentials[lower] * self.values[lower] as f64
                + exponentials[upper] * self.values[upper] as f64)
                / normalizer;
        }
        value as f32
    }
}

pub fn softmax_unimix(logits: &[f32], unimix: f32, output: &mut [f32]) {
    assert_eq!(logits.len(), output.len());
    assert!(!logits.is_empty());
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>();
    let uniform = unimix / logits.len() as f32;
    for (probability, logit) in output.iter_mut().zip(logits) {
        *probability = (1.0 - unimix) * (*logit - maximum).exp() / normalizer + uniform;
    }
}

pub fn sample_probabilities(probabilities: &[f32], rng: &mut impl Rng) -> usize {
    assert!(!probabilities.is_empty());
    assert!(
        probabilities
            .iter()
            .all(|probability| probability.is_finite() && *probability >= 0.0)
    );
    let total = probabilities.iter().sum::<f32>();
    assert!(total > 0.0);
    let draw = rng.random::<f32>() * total;
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += *probability;
        if draw < cumulative {
            return index;
        }
    }
    probabilities
        .iter()
        .rposition(|probability| *probability > 0.0)
        .expect("positive probability mass")
}

pub fn sample_logits(logits: &[f32], unimix: f32, rng: &mut impl Rng) -> usize {
    let mut probabilities = vec![0.0; logits.len()];
    softmax_unimix(logits, unimix, &mut probabilities);
    sample_probabilities(&probabilities, rng)
}

/// Exact alignment used by the pinned D3 `lambda_return` helper.
/// Every input has length `N`; the output has length `N - 1` and uses
/// reward/termination from indices `1..N`.
pub fn lambda_returns(
    last: &[f32],
    terminal: &[f32],
    reward: &[f32],
    value: &[f32],
    bootstrap: &[f32],
    discount: f32,
    lambda: f32,
) -> Vec<f32> {
    let length = last.len();
    assert!(length >= 2);
    assert_eq!(terminal.len(), length);
    assert_eq!(reward.len(), length);
    assert_eq!(value.len(), length);
    assert_eq!(bootstrap.len(), length);
    let mut result = vec![0.0; length - 1];
    let mut next = bootstrap[length - 1];
    for time in (0..length - 1).rev() {
        let index = time + 1;
        let live = (1.0 - terminal[index]) * discount;
        let trace = (1.0 - last[index]) * lambda;
        let intermediate = reward[index] + (1.0 - trace) * live * bootstrap[index];
        next = intermediate + live * trace * next;
        result[time] = next;
    }
    // `value` is asserted/aligned because callers commonly pass it as both
    // the lambda target and bootstrap. Keep the argument explicit to prevent
    // accidental one-step shifts at call sites.
    let _ = value;
    result
}

#[derive(Clone, Debug)]
pub struct PercentileNormalizer {
    low: f32,
    high: f32,
    rate: f32,
    minimum_scale: f32,
}

impl PercentileNormalizer {
    pub fn new(rate: f32, minimum_scale: f32) -> Self {
        assert!((0.0..=1.0).contains(&rate));
        assert!(minimum_scale > 0.0);
        Self {
            low: 0.0,
            high: 0.0,
            rate,
            minimum_scale,
        }
    }

    pub fn update(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f32::total_cmp);
        let percentile = |q: f32| {
            let position = q * (sorted.len() - 1) as f32;
            let low = position.floor() as usize;
            let high = position.ceil() as usize;
            let mix = position - low as f32;
            sorted[low] * (1.0 - mix) + sorted[high] * mix
        };
        self.low = (1.0 - self.rate) * self.low + self.rate * percentile(0.05);
        self.high = (1.0 - self.rate) * self.high + self.rate * percentile(0.95);
    }

    pub fn scale(&self) -> f32 {
        (self.high - self.low).max(self.minimum_scale)
    }

    pub(crate) fn state(&self) -> (f32, f32) {
        (self.low, self.high)
    }

    pub(crate) fn restore(&mut self, low: f32, high: f32) {
        assert!(low.is_finite() && high.is_finite());
        self.low = low;
        self.high = high;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn two_hot_uses_exponential_raw_value_bins() {
        let bins = TwoHotBins::new(255);
        assert_eq!(bins.values()[127], 0.0);
        assert_eq!(bins.values()[0], -bins.values()[254]);
        assert_eq!(bins.decode_logits(&vec![0.0; bins.len()]), 0.0);
        let mut target = vec![0.0; bins.len()];
        bins.encode(1_000.0, &mut target);
        assert!((target.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(target.iter().filter(|weight| **weight > 0.0).count() <= 2);
    }

    #[test]
    fn unimix_keeps_every_class_live() {
        let mut probabilities = [0.0; 4];
        softmax_unimix(
            &[1_000.0, -1_000.0, -1_000.0, -1_000.0],
            0.01,
            &mut probabilities,
        );
        assert!(
            probabilities
                .iter()
                .all(|probability| *probability >= 0.0025)
        );
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let mut rng = StdRng::seed_from_u64(7);
        assert!(sample_probabilities(&probabilities, &mut rng) < 4);
        for _ in 0..100 {
            assert_eq!(sample_probabilities(&[0.0, 1.0, 0.0], &mut rng), 1);
        }
    }

    #[test]
    fn lambda_return_matches_d3_indexing() {
        let result = lambda_returns(
            &[0.0, 0.0, 1.0],
            &[0.0, 0.0, 1.0],
            &[99.0, 1.0, 2.0],
            &[10.0, 20.0, 30.0],
            &[10.0, 20.0, 30.0],
            0.9,
            1.0,
        );
        assert_eq!(result, vec![2.8, 2.0]);
    }
}
