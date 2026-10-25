//! Full-model target-state checks outside the unchanged complete-update timer.
use super::*;

fn normalizer(core: &mut DreamerCore) -> [u32; 2] {
    let (low, high) = core.return_normalizer.state();
    let bits = [low.to_bits(), high.to_bits()];
    if let Some(targets) = &mut core.targets_gpu {
        assert_eq!(targets.normalizer_state().map(f32::to_bits), bits);
    }
    bits
}

pub(super) fn initial_state(core: &mut DreamerCore, root: &Path) {
    let bits = normalizer(core);
    save_json(
        &root.join("target-initial-state.json"),
        &serde_json::json!({
            "normalizer_bits": bits,
            "device_targets": core.targets_gpu.is_some(),
            "learner_step": core.learner_step,
            "environment_step": core.environment_step,
        }),
    );
}

pub(super) fn checkpoint_roundtrip(mut core: DreamerCore, root: &Path) {
    let before = normalizer(&mut core);
    let steps = (core.learner_step, core.environment_step);
    let optimizer_steps = (
        core.world_train.adam_step_count(),
        core.behavior_train.adam_step_count(),
    );
    let resident = core.targets_gpu.is_some();
    let saved = root.join("learner-checkpoint");
    core.save_checkpoint(&saved).unwrap();
    let metadata = read_checkpoint_metadata(&saved).unwrap();
    assert_eq!(
        [
            metadata.return_low.to_bits(),
            metadata.return_high.to_bits()
        ],
        before
    );
    let gpu = Arc::clone(&core.gpu);
    drop(core);
    let mut restored = DreamerCore::restore_with_gpu(&saved, gpu, metadata).unwrap();
    assert_eq!(normalizer(&mut restored), before);
    assert_eq!((restored.learner_step, restored.environment_step), steps);
    assert_eq!(
        (
            restored.world_train.adam_step_count(),
            restored.behavior_train.adam_step_count(),
        ),
        optimizer_steps
    );
    assert_eq!(restored.targets_gpu.is_some(), resident);
    let memory = restored.gpu_memory_budget();
    let headroom = memory.budget_bytes.saturating_sub(memory.usage_bytes);
    assert!(headroom >= 2 << 30);
    checkpoint(&mut restored, &root.join("restored"));
    save_json(
        &root.join("target-checkpoint.json"),
        &serde_json::json!({
            "status": "complete", "normalizer_bits": before,
            "device_targets": resident, "learner_step": steps.0,
            "environment_step": steps.1, "optimizer_steps": optimizer_steps,
            "minimum_sampled_headroom_bytes": headroom,
            "extra_learner_updates": 0, "game_actions": 0,
            "independent_tensor_review_pending": true,
        }),
    );
}
