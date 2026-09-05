import kindle


def test_pinned_baseline_and_d3_defaults() -> None:
    config = kindle.default_config(18, "12m")
    assert config["action_count"] == 18
    assert config["model_size"] == "size12_m"
    assert config["observation_decoder_depth"] == 64
    assert config["batch_size"] == 16
    assert config["batch_length"] == 64
    assert config["world_backprop_length"] == 8
    assert config["world_microbatch_size"] is None
    assert config["imagination_length"] == 15
    assert config["actor_unimix"] == 0.01
    assert config["train_ratio"] == 32.0
    assert config["loss_scales"]["reconstruction"] == 1.0
    assert config["loss_scales"]["future_prediction"] == 0.0
    assert config["visitation_bonus"] is False
    assert config["intrinsic_reward_scale"] == 0.0
    assert config["replay_value_gradient"] is True
    assert config["behavior_learning_rate"] is None
    assert config["actor_learning_starts"] == 0
    assert config["dynamics_free_nats"] is None
    assert kindle.DINO_MODEL_ID.startswith("facebook/dinov3-vits16")


def test_unknown_size_is_rejected() -> None:
    try:
        kindle.default_config(4, "large-ish")
    except ValueError as error:
        assert "unknown model_size" in str(error)
    else:
        raise AssertionError("invalid model size was accepted")


def test_invalid_action_count_is_a_python_error() -> None:
    try:
        kindle.default_config(1)
    except ValueError as error:
        assert "num_actions must be greater than one" in str(error)
    else:
        raise AssertionError("invalid action count was accepted")


def test_agent_exposes_read_only_reward_probes() -> None:
    assert hasattr(kindle.Agent, "posterior_reward_prediction")
    assert hasattr(kindle.Agent, "prior_reward_prediction")


def test_agent_exposes_read_only_behavior_probes() -> None:
    assert hasattr(kindle.Agent, "posterior_value_prediction")
    assert hasattr(kindle.Agent, "posterior_action_probabilities")
    assert hasattr(kindle.Agent, "prior_behavior_rollout")


def test_agent_exposes_read_only_dynamics_probes() -> None:
    assert hasattr(kindle.Agent, "dino_observation")
    assert hasattr(kindle.Agent, "latent_feature")
    assert hasattr(kindle.Agent, "observation_prediction")
    assert hasattr(kindle.Agent, "prior_diagnostic_rollout")
    assert hasattr(kindle.Agent, "provenance")
