"""Error-path coverage for the pyo3 bindings.

Every case here used to cross the FFI boundary as a Rust panic
(pyo3 PanicException, a BaseException) or silently mis-train; the
bindings now validate up front and raise ValueError (audit M-14,
M-15, M-19). Kept GPU-light: constructors plus a step or two.
"""

import pytest

import kindle


def test_num_actions_over_max_raises_value_error():
    # MAX_ACTION_DIM is exported by the extension.
    with pytest.raises(ValueError, match="num_actions"):
        kindle.Agent(obs_dim=4, num_actions=100)


def test_batch_num_actions_over_max_raises_value_error():
    with pytest.raises(ValueError, match="num_actions"):
        kindle.BatchAgent(
            obs_dim=4,
            num_actions=kindle.MAX_ACTION_DIM + 1,
            batch_size=1,
        )


def test_obs_dim_over_token_dim_raises_value_error():
    with pytest.raises(ValueError, match="obs_dim"):
        kindle.Agent(obs_dim=kindle.OBS_TOKEN_DIM + 1, num_actions=2)


def test_agent_wrong_obs_length_raises():
    agent = kindle.Agent(obs_dim=4, num_actions=2, env_id=0, seed=0)
    # Too short (would silently zero-pad before the fix).
    with pytest.raises(ValueError, match="obs_dim"):
        agent.act([0.0, 0.0])
    # Too long (would silently truncate).
    with pytest.raises(ValueError, match="obs_dim"):
        agent.act([0.0] * 5)
    # Correct length works, including in observe.
    action = agent.act([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="obs_dim"):
        agent.observe([0.0, 0.0, 0.0], action)
    agent.observe([0.0, 0.1, 0.0, 0.1], action)
    assert agent.step_count() == 1


def test_batch_agent_wrong_obs_length_raises():
    agent = kindle.BatchAgent(obs_dim=4, num_actions=2, batch_size=2, seed=0)
    with pytest.raises(ValueError, match="obs_dim"):
        agent.act([[0.0] * 4, [0.0] * 3])
    actions = agent.act([[0.0] * 4, [0.0] * 4])
    with pytest.raises(ValueError, match="obs_dim"):
        agent.observe([[0.0] * 4, [0.0] * 5], actions)


def test_batch_agent_wrong_homeo_length_raises():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=2, seed=0)
    obs = [[0.0, 0.0], [0.0, 0.0]]
    actions = agent.act(obs)
    homeo = [(0.1, 0.0, 0.05)]
    # Too short: used to be silently padded with empty lists (M-19).
    with pytest.raises(ValueError, match="batch_size"):
        agent.observe(obs, actions, homeostatic=[[homeo[0]]])
    # Too long.
    with pytest.raises(ValueError, match="batch_size"):
        agent.observe(obs, actions, homeostatic=[[], [], []])
    # Exact length works.
    agent.observe(obs, actions, homeostatic=[[homeo[0]], []])
    assert agent.step_count() == 1


def test_batch_agent_setter_length_checks():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=2, seed=0)
    # set_extrinsic_reward expects batch_size entries — used to panic
    # (or silently no-op under the alpha gate) on mismatch (M-14).
    with pytest.raises(ValueError, match="set_extrinsic_reward"):
        agent.set_extrinsic_reward([1.0])
    with pytest.raises(ValueError, match="set_extrinsic_reward"):
        agent.set_extrinsic_reward([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="set_extrinsic_events"):
        agent.set_extrinsic_events([True])
    with pytest.raises(ValueError, match="set_intrinsic_progress"):
        agent.set_intrinsic_progress([1.0])
    # set_action_masks expects batch_size * MAX_ACTION_DIM floats.
    with pytest.raises(ValueError, match="set_action_masks"):
        agent.set_action_masks([1.0] * 5)
    with pytest.raises(ValueError, match="set_action_parameter_masks"):
        agent.set_action_parameter_masks([False] * 5)
    with pytest.raises(ValueError, match="train_coord_head active_mask"):
        agent.train_coord_head([True])
    with pytest.raises(ValueError, match="train_coord_head rewards"):
        agent.train_coord_head([True, False], rewards=[1.0])
    with pytest.raises(ValueError, match="set_action_parameters"):
        agent.set_action_parameters([0.0, 0.0], [True, False])
    with pytest.raises(ValueError, match="set_action_parameters active_mask"):
        agent.set_action_parameters([0.0] * 4, [True])
    # Correct lengths are accepted.
    agent.set_extrinsic_reward([0.0, 0.0])
    agent.set_extrinsic_events([False, True])
    agent.set_intrinsic_progress([0.0, 0.0])
    agent.set_action_masks([1.0] * (2 * kindle.MAX_ACTION_DIM))
    agent.set_action_parameter_masks([False] * (2 * kindle.MAX_ACTION_DIM))
    agent.set_action_parameters([0.0] * 4, [True, False])
    agent.train_coord_head([True, False], rewards=[1.0, 0.0])


def test_action_parameters_are_per_lane_clamped_and_one_shot():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=2, seed=0)
    obs = [[0.0, 0.0], [0.0, 0.0]]

    actions = agent.act(obs)
    agent.set_action_parameters([-0.5, 0.75, 3.0, -3.0], [True, False])
    agent.observe(obs, actions)
    assert agent.dump_buffer(0, 1)[0]["action_parameters"] == pytest.approx(
        [-0.5, 0.75]
    )
    assert agent.dump_buffer(1, 1)[0]["action_parameters"] == pytest.approx([0.0, 0.0])

    # Staging is consumed by observe(); an ordinary next action cannot
    # accidentally inherit the preceding click coordinates.
    actions = agent.act(obs)
    agent.observe(obs, actions)
    assert agent.dump_buffer(0, 1)[0]["action_parameters"] == pytest.approx([0.0, 0.0])


def test_parameterized_queue_survives_act_and_is_consumed_once():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=1, seed=0)
    mask = [False] * kindle.MAX_ACTION_DIM
    mask[1] = True
    agent.set_action_parameter_masks(mask)
    agent.queue_parameterized_actions(0, [(1, (2.0, -2.0))])

    assert agent.act([[0.0, 0.0]]) == [1]
    assert agent.take_planned_action_parameters() == [(1.0, -1.0)]
    assert agent.take_planned_action_parameters() == [None]


def test_planner_queue_rejects_action_invalidated_by_dynamic_mask():
    agent = kindle.BatchAgent(obs_dim=2, num_actions=2, batch_size=1, seed=0)
    availability = [0.0] * kindle.MAX_ACTION_DIM
    availability[0] = 1.0
    agent.set_action_masks(availability)
    parameter_mask = [False] * kindle.MAX_ACTION_DIM
    parameter_mask[1] = True
    agent.set_action_parameter_masks(parameter_mask)
    agent.queue_parameterized_actions(0, [(1, (0.25, -0.5))])

    assert agent.act([[0.0, 0.0]]) == [0]
    assert agent.take_planned_action_parameters() == [None]
    assert agent.planner_queue_len() == 0


def test_supervised_coordinate_imitation_and_persistence(tmp_path):
    kwargs = dict(
        obs_dim=4,
        num_actions=2,
        batch_size=2,
        env_ids=[7, 7],
        seed=9,
        latent_dim=16,
        hidden_dim=32,
        coord_action_alpha=1.0,
        coord_hidden_dim=16,
        coord_lr=0.05,
    )
    agent = kindle.BatchAgent(**kwargs)
    observations = [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]]
    targets = [(-0.7, 0.4), (0.7, -0.4)]

    with pytest.raises(ValueError, match="targets"):
        agent.train_coord_head_supervised(observations, targets[:1])
    with pytest.raises(ValueError, match="active_mask"):
        agent.train_coord_head_supervised(
            observations, targets, active_mask=[True]
        )
    with pytest.raises(ValueError, match="weight"):
        agent.train_coord_head_supervised(observations, targets, weight=-1.0)

    initial = agent.coord_means(observations)
    for _ in range(200):
        agent.train_coord_head_supervised(observations, targets)
    trained = agent.coord_means(observations)

    def squared_error(predictions):
        return sum(
            (px - tx) ** 2 + (py - ty) ** 2
            for (px, py), (tx, ty) in zip(predictions, targets)
        )

    assert squared_error(trained) < squared_error(initial) * 0.1

    checkpoint = tmp_path / "weights"
    agent.save_weights(str(checkpoint))
    assert (checkpoint / "coord.bin").is_file()
    loaded = kindle.BatchAgent(**kwargs)
    loaded.load_weights(str(checkpoint))
    assert loaded.coord_means(observations) == pytest.approx(trained, abs=1e-6)


def test_coordinate_head_uses_task_context_across_batch_sizes(tmp_path):
    common = dict(
        obs_dim=4,
        num_actions=2,
        seed=21,
        latent_dim=16,
        hidden_dim=32,
        coord_action_alpha=1.0,
        coord_hidden_dim=32,
        coord_lr=0.01,
    )
    observation = [0.25, -0.5, 0.75, 0.0]
    observations = [observation, observation]
    targets = [(-0.8, 0.6), (0.7, -0.5)]
    agent = kindle.BatchAgent(batch_size=2, env_ids=[101, 202], **common)

    for _ in range(500):
        agent.train_coord_head_supervised(observations, targets)
    trained = agent.coord_means(observations)
    for prediction, target in zip(trained, targets):
        assert prediction == pytest.approx(target, abs=0.05)

    checkpoint = tmp_path / "task-conditioned-coordinates"
    agent.save_weights(str(checkpoint))
    for env_id, target in zip((101, 202), targets):
        loaded = kindle.BatchAgent(batch_size=1, env_ids=[env_id], **common)
        loaded.load_weights(str(checkpoint))
        assert loaded.coord_means([observation])[0] == pytest.approx(
            target, abs=0.05,
        )


def test_supervised_policy_imitation_masks_and_cross_batch_reload(tmp_path):
    common = dict(
        obs_dim=4,
        num_actions=3,
        seed=17,
        latent_dim=16,
        hidden_dim=32,
        learning_rate=0.0,
        lr_policy=0.001,
        end_to_end_encoder=True,
        n_step=2,
        rollout_length=2,
        value_loss_coef=0.0,
        recon_loss_coef=0.0,
        entropy_beta=0.0,
    )
    agent = kindle.BatchAgent(batch_size=2, env_ids=[7, 8], **common)
    observations = [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]]
    labels = [1, 2]
    masks = [0.0] * (2 * kindle.MAX_ACTION_DIM)
    masks[0] = masks[1] = 1.0
    masks[kindle.MAX_ACTION_DIM + 1] = 1.0
    masks[kindle.MAX_ACTION_DIM + 2] = 1.0
    agent.set_action_masks(masks)

    with pytest.raises(ValueError, match="action_indices length"):
        agent.train_policy_supervised(observations, [1])
    with pytest.raises(ValueError, match="outside num_actions"):
        agent.train_policy_supervised(observations, [1, 3])
    with pytest.raises(ValueError, match="weight"):
        agent.train_policy_supervised(observations, labels, weight=0.0)
    with pytest.raises(ValueError, match="masked on lane 0"):
        agent.train_policy_supervised(observations, [2, 2])

    for _ in range(500):
        agent.train_policy_supervised(observations, labels)
    assert agent.act(observations, deterministic=True) == labels

    checkpoint = tmp_path / "policy-weights"
    agent.save_weights(str(checkpoint))
    loaded = kindle.BatchAgent(batch_size=1, env_ids=[7], **common)
    loaded.set_action_masks(masks[:kindle.MAX_ACTION_DIM])
    loaded.load_weights(str(checkpoint))
    assert loaded.act([observations[0]], deterministic=True) == [labels[0]]
