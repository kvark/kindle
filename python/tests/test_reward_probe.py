from kindle._reward_probe import RewardProbe, roc_auc


def test_roc_auc_handles_ordering_ties_and_missing_classes() -> None:
    assert roc_auc([False, True], [0.0, 1.0]) == 1.0
    assert roc_auc([False, True], [1.0, 0.0]) == 0.0
    assert roc_auc([False, True], [0.0, 0.0]) == 0.5
    assert roc_auc([True], [1.0]) is None


def test_reward_probe_summarizes_calibration_and_ranking() -> None:
    probe = RewardProbe()
    probe.record(-1.0, -0.8, -0.9)
    probe.record(0.0, 0.0, 0.0)
    probe.record(1.0, 0.8, 0.9)

    summary = probe.summary()
    assert summary["samples"] == 3
    assert summary["by_reward_sign"]["negative"]["count"] == 1
    assert summary["by_reward_sign"]["zero"]["count"] == 1
    assert summary["by_reward_sign"]["positive"]["count"] == 1
    assert summary["one_step_prior_ranking"] == {
        "positive_vs_rest_roc_auc": 1.0,
        "negative_vs_rest_roc_auc": 1.0,
        "reward_event_vs_zero_roc_auc": 1.0,
        "positive_vs_negative_roc_auc": 1.0,
    }
    assert summary["posterior_ranking"] == {
        "positive_vs_rest_roc_auc": 1.0,
        "negative_vs_rest_roc_auc": 1.0,
        "reward_event_vs_zero_roc_auc": 1.0,
        "positive_vs_negative_roc_auc": 1.0,
    }
