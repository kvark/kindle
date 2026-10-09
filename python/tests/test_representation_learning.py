import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from summarize_representation_learning import TINY_CHECKPOINTS, markdown, plot_svg, read_run, summarize


def fixture(seed=1009, *, upstream=False, ticks=4, reward=1., replication=False):
    n = 8 if replication else 6
    config = (dict(batch_size=8, batch_length=16, train_ratio=32) if replication else
              dict(batch_size=16, batch_length=64, train_ratio=256))
    start = dict(event="run_start", seed=seed, num_envs=n, steps=n*ticks,
                 environment_seeds=[seed+i*1000003 for i in range(n)], full_action_space=True,
                 sticky_actions=.25, action_repeat=4, noop_max=0, max_episode_frames=100000)
    if upstream:
        start.update(game="pong", protocol="phase2-matched-actions-v1", observation_size=64,
                     reward_action_aids="none", compute_dtype="float32", **config)
    else:
        start.update(environment="ALE/Pong-v5", config=dict(seed=seed, **config), mode="train",
                     starting_environment_step=0, starting_learner_step=0, restored_checkpoint=None,
                     observation_size="native")
    if replication:
        if upstream:
            start.update(game="seaquest", protocol="replication-matched-actions-v2",
                         model_size="1m", replay_context=1, replay_arrival_capacity=100000)
        else:
            start.update(environment="ALE/Seaquest-v5", learned_rgb_preprocessing="GPU Pillow-equivalent",
                         model_provenance=dict(perception=None))
            start["config"].update(model_size="size1_m", observation_kind="rgb64", actor_unimix=0,
                world_backprop_length=16, world_microbatch_size=8, imagination_length=15,
                replay_capacity=100000, replay_context=1, intrinsic_reward_scale=0, visitation_bonus=False,
                loss_scales=dict(reconstruction=1., future_prediction=0.))
    rows, returns, lengths, counts = [start], [0.] * n, [0] * n, [0] * n
    for tick in range(1, ticks+1):
        done = tick % 3 == 0
        rows.append(dict(event="transition", run_step=n*tick, actions=[0]*n, rewards=[reward]*n,
                         terminated=[done]*n, truncated=[False]*n))
        updates = 1 + (n*tick-n)//4
        rows.append(dict(event="learner", run_step=n*tick))
        for i in range(n):
            returns[i] += reward
            lengths[i] += 1
            if done:
                rows.append(dict(event="episode", stream=i, episode=counts[i], run_step=n*tick,
                                 episode_return=returns[i], episode_length=lengths[i],
                                 terminated=True, truncated=False, elapsed_seconds=float(tick)))
                counts[i] += 1
                returns[i], lengths[i] = 0., 0
        rows.append(dict(event="progress", run_step=n*tick, learner_step=updates, elapsed_seconds=float(tick),
                         total_rewards=[reward*tick]*n, episode_counts=counts.copy(),
                         partial_returns=returns.copy(), partial_lengths=lengths.copy()))
    rows.append(rows[-1] | dict(event="run_end", reason="budget_complete", learner_updates=updates,
                                elapsed_seconds=float(ticks)+.5))
    return rows


def write(tmp_path, rows, name="log.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.mark.parametrize("upstream", [False, True])
def test_learning_reader_reconciles_complete_episodes_and_partial_tails(tmp_path, upstream):
    path = write(tmp_path, fixture(upstream=upstream))
    run = read_run(path)
    assert len(run["episodes"]) == 6
    assert run["final"]["partial_returns"] == [1.]*6
    assert run["final"]["partial_lengths"] == [1]*6
    assert run["curve"][-1]["score"] == 3
    assert [p["actions"] for p in run["curve"]] == [6, 12, 18, 24]
    assert run["curve"][-1]["seconds"] == 4.5
    assert run["curve"][0]["score"] is None
    assert len(run["source_sha256"]) == 64 and run["first_training_action"] == 6


def test_learning_summary_requires_pixels_and_joint_loss_for_cnn_label(tmp_path):
    rows = fixture()
    with pytest.raises(ValueError, match="jointly learned"):
        summarize([("learned_cnn", write(tmp_path, rows))], budget=24)
    rows[0]["config"].update(observation_kind="rgb64", loss_scales=dict(reconstruction=1., future_prediction=0.))
    rows[0].update(model_provenance=dict(perception=None), learned_rgb_preprocessing="single GPU resize")
    path = write(tmp_path, rows)
    assert summarize([("learned_cnn", path)], budget=24)["results"][0]["method"] == "learned_cnn"
    with pytest.raises(ValueError, match="labelled frozen"):
        summarize([("pretrained_tiny", path)], budget=24)
    rows[0]["model_provenance"]["perception"] = dict(kind="levjepa-tiny")
    with pytest.raises(ValueError, match="jointly learned"):
        summarize([("learned_cnn", write(tmp_path, rows))], budget=24)


@pytest.mark.parametrize("corruption", ["missing_end", "missing_transition", "missing_episode", "reward", "count", "time", "updates", "nan", "extra_end"])
def test_learning_reader_rejects_incomplete_or_inconsistent_logs(tmp_path, corruption):
    rows = fixture()
    if corruption == "missing_end":
        rows.pop()
    elif corruption == "missing_transition":
        del rows[1]
    elif corruption == "missing_episode":
        rows.pop(next(i for i, row in enumerate(rows) if row["event"] == "episode"))
    elif corruption == "reward":
        next(row for row in rows if row["event"] == "episode")["episode_return"] = 10
    elif corruption == "count":
        rows[-1]["episode_counts"] = [2]*6
    elif corruption == "time":
        rows[-1]["elapsed_seconds"] = -1
    elif corruption == "updates":
        rows[-1].update(learner_step=6, learner_updates=6)
    elif corruption == "nan":
        rows[-1]["elapsed_seconds"] = float("nan")
    else:
        rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError):
        read_run(write(tmp_path, rows))


def test_learning_summary_uses_seed_not_episode_bootstrap_and_keeps_failures(tmp_path):
    inputs = []
    for seed, reward in [(1009, -1.), (2017, 0.), (3019, 2.)]:
        inputs.append(("pretrained_tiny", write(tmp_path, fixture(seed, reward=reward), f"{seed}.jsonl")))
    result = summarize(inputs, budget=24)
    row = result["results"][0]
    assert result["status"] == "partial_learning_comparison" and not result["phase2_complete"]
    assert row["aggregate"]["final_score"] == dict(mean=1., ci95=[-3., 6.], seeds=[-3., 0., 6.])
    assert row["aggregate"]["final_hns"]["mean"] == pytest.approx((1+20.7)/35.3)
    assert row["aggregate"]["by_time"][0]["seconds"] == 3
    assert len(row["runs"]) == 3 and all(len(run["episodes"]) == 6 for run in row["runs"])
    assert "Human-normalized" in markdown(result, "data.json")
    partial = summarize(inputs[:2], budget=24)
    assert partial["results"][0]["aggregate"] is None
    assert "1009: -3.000; 2017: 0.000" in markdown(partial, "partial.json")
    assert "no aggregate or uncertainty" in markdown(partial, "partial.json")
    with pytest.raises(ValueError, match="duplicate"):
        summarize(inputs + inputs[:1], budget=24)
    with pytest.raises(ValueError, match="budget"):
        summarize(inputs)


def test_partial_learning_report_does_not_fabricate_a_score_before_any_episode(tmp_path):
    result = summarize([("upstream", write(tmp_path, fixture(upstream=True, ticks=2)))], budget=12)
    assert "1009: no completed episode" in markdown(result, "partial.json")


def test_learning_curve_keeps_all_episodes_but_scores_only_last_fifty(tmp_path):
    rows = fixture(ticks=30)
    # Make the first six episodes different, retaining a consistent raw ledger.
    for row in rows:
        if row["event"] == "transition" and row["run_step"] <= 18:
            row["rewards"] = [-1.]*6
        elif row["event"] == "episode" and row["episode"] == 0:
            row["episode_return"] = -3.
        elif row["event"] in ("progress", "run_end"):
            tick = row["run_step"] // 6
            row["total_rewards"] = [tick - 2*min(tick, 3)]*6
            if tick < 3:
                row["partial_returns"] = [-tick]*6
    run = read_run(write(tmp_path, rows))
    assert len(run["episodes"]) == 60 and run["curve"][-1]["completed_episodes"] == 60
    assert run["episodes"][0]["episode_return"] == -3
    assert run["curve"][-1]["score"] == 3


def test_upstream_delayed_metrics_do_not_shift_first_update_accounting(tmp_path):
    rows = fixture(upstream=True)
    del rows[2]  # No metric returned by the first asynchronous train call.
    rows[-1]["first_training_action"] = 6
    run = read_run(write(tmp_path, rows))
    assert run["first_training_action"] == 6


def test_learning_plot_shows_partial_seed_traces_without_fabricated_zeros(tmp_path):
    result = summarize([("upstream", write(tmp_path, fixture(upstream=True)))], budget=24)
    root = ET.fromstring(plot_svg(result))
    ns = {"s": "http://www.w3.org/2000/svg"}
    assert not root.findall(".//s:polygon", ns)
    traces = root.findall(".//s:g", ns)
    assert len(traces) == 2 and all(g.attrib["data-seed"] == "1009" for g in traces)
    for trace in traces:
        line = trace.find("s:polyline", ns)
        assert line.attrib["stroke-dasharray"] == "8 3"
        assert len(line.attrib["points"].split()) == 2  # The first two scores are missing, not zero.
    # Even an entirely unscored report is a valid, explicitly empty chart.
    result["results"][0]["runs"][0]["curve"] = [dict(actions=6, seconds=1., score=None)]
    empty = ET.fromstring(plot_svg(result))
    assert not empty.findall(".//s:polyline", ns)
    assert "No completed-episode scores yet" in " ".join(empty.itertext())


def test_learning_plot_uses_aggregate_values_and_common_time_support(tmp_path):
    inputs = [("pretrained_tiny", write(tmp_path, fixture(seed, reward=i), f"{seed}.jsonl"))
              for i, seed in enumerate((1009, 2017, 3019))]
    result = summarize(inputs, budget=24)
    aggregate = result["results"][0]["aggregate"]
    aggregate["by_time"] = aggregate["by_time"][:5]  # Renderer must not extend to the individual tails.
    root = ET.fromstring(plot_svg(result))
    ns = {"s": "http://www.w3.org/2000/svg"}
    traces = root.findall(".//s:g", ns)
    assert len(traces) == 2 and all(g.attrib["data-seed"] == "aggregate" for g in traces)
    for trace, points in zip(traces, (aggregate["by_actions"], aggregate["by_time"])):
        line = trace.find("s:polyline", ns)
        assert line.attrib["stroke-dasharray"] == "none"
        assert len(line.attrib["points"].split()) == len(points)
        assert len(trace.find("s:polygon", ns).attrib["points"].split()) == 2*len(points)
    aggregate["by_time"][0]["score"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        plot_svg(result)


@pytest.mark.parametrize("upstream", [False, True])
def test_small_replication_is_explicit_and_keeps_stream_and_recipe_identity(tmp_path, upstream):
    rows = fixture(upstream=upstream, replication=True)
    method = "upstream" if upstream else "learned_cnn"
    path = write(tmp_path, rows)
    result = summarize([(method, path)], budget=32, replication=True)
    assert result["num_envs"] == 8 and result["games"] == ("Seaquest",)
    assert result["status"] == "partial_learning_comparison" and not result["phase2_complete"]
    assert len(result["results"][0]["runs"][0]["episodes"]) == 8
    svg = plot_svg(result)
    assert "8 streams combined" in svg and "Seaquest · 1/6 runs" in svg and "Native Dreamer" in svg
    assert "Pong" not in svg and "Breakout" not in svg
    with pytest.raises(ValueError, match="budget"):
        summarize([(method, path)], budget=32)
    if upstream:
        rows[0]["model_size"] = "12m"
    else:
        rows[0]["config"]["actor_unimix"] = .01
    with pytest.raises(ValueError, match="Size1M"):
        summarize([(method, write(tmp_path, rows))], budget=32, replication=True)


def test_small_replication_requires_six_runs_not_the_cancelled_matrix(tmp_path):
    inputs = []
    for seed in (1009, 2017, 3019):
        for upstream, method in [(True, "upstream"), (False, "learned_cnn")]:
            rows = fixture(seed, upstream=upstream, replication=True)
            inputs.append((method, write(tmp_path, rows, f"{method}-{seed}.jsonl")))
    result = summarize(inputs, budget=32, replication=True)
    assert result["status"] == "replication_complete" and not result["phase2_complete"]
    assert len(result["results"]) == 2 and all(r["aggregate"] for r in result["results"])
    assert result["paired_scores"][0]["difference"] == dict(mean=0., ci95=[0., 0.], seeds=[0., 0., 0.])
    assert "does not test JEPA" in " ".join(result["limits"])
    with pytest.raises(ValueError, match="unknown method"):
        summarize([("large", inputs[0][1])], budget=32, replication=True)


def test_interrupted_prefix_is_audited_but_never_counted_as_a_complete_run(tmp_path):
    rows = fixture(replication=True)
    rows[0]["steps"] = 200000
    rows[-1]["reason"] = "interrupted"
    path = write(tmp_path, rows)
    with pytest.raises(ValueError, match="incomplete"):
        read_run(path)
    run = read_run(path, allow_interrupted=True)
    assert not run["complete"] and run["final"]["run_step"] == 32
    assert len(run["episodes"]) == 8 and run["final"]["partial_lengths"] == [1]*8
    plot = dict(results=[], interrupted_runs=[dict(method="learned_cnn", game="Seaquest", run=run)],
                methods=("upstream", "learned_cnn"), games=("Seaquest",), num_envs=8,
                comparison="small_replication", action_budget=200000)
    assert "0/6 complete + 1 interrupted" in plot_svg(plot)
    with pytest.raises(ValueError, match="incomplete"):
        summarize([("learned_cnn", path)], budget=200000, replication=True)
    rows[-1]["total_rewards"][0] = -999
    with pytest.raises(ValueError, match="transitions"):
        read_run(write(tmp_path, rows), allow_interrupted=True)


def tiny_fixture(seed, method, *, reward=1.):
    rows = fixture(seed, replication=True, reward=reward)
    rows[0]["config"].update(observation_kind="features", loss_scales=dict(reconstruction=0., future_prediction=.25))
    rows[0]["model_provenance"]["perception"] = dict(kind="levjepa-tiny", checkpoint_sha256=TINY_CHECKPOINTS[method])
    del rows[0]["learned_rgb_preprocessing"]
    return rows


def joint_fixture(seed, mode, *, reward=1.):
    rows = tiny_fixture(seed, "pretrained_tiny", reward=reward)
    rows[0]["config"].update(video_encoder=mode, world_microbatch_size=1, replay_capacity=8192,
                            learning_rate=4e-5, learning_rate_warmup=1000, agc=.3, replay_value_gradient=True)
    for row in rows:
        if row["event"] == "learner":
            row["report"] = dict(world=dict(encoder_spread=.2, future_prediction_loss=3.),
                                 behavior=dict(policy_entropy=2.), timing=dict(total_seconds=1.))
    return rows


def test_joint_tiny_summary_pairs_audited_seeds_and_keeps_diagnostics(tmp_path, monkeypatch):
    # Scheduler/phase/reset/eviction auditing is exercised with complete ledgers
    # in test_vector.py; these compact fixtures isolate summary validation.
    checked = []
    monkeypatch.setattr("kindle._vector_audit.audit", lambda path: checked.append(path) or dict(accounting_valid=True))
    inputs = [(f"{mode}_tiny", write(tmp_path, joint_fixture(seed, mode, reward=i + (mode == "joint")), f"{mode}-{seed}.jsonl"))
              for i, seed in enumerate((1009, 2017, 3019)) for mode in ("frozen", "joint")]
    result = summarize(inputs, budget=32, joint_tiny=True)
    assert len(checked) == 6 and result["status"] == "learning_comparison_complete"
    assert result["paired_scores"][0]["difference"] == dict(mean=3., ci95=[3., 3.], seeds=[3.]*3)
    for group in result["results"]:
        for run in group["runs"]:
            assert run["accounting_audit"]["accounting_valid"]
            point = run["curve"][-1]
            assert point["reported_updates"] == 1 and point["seconds"] == 4.5
            assert point["learner_mean"]["world"]["encoder_spread"] == .2
    assert "Joint Tiny · online" in plot_svg(result)
    assert "not direct actor-loss gradients" in markdown(result, "data.json")
    assert summarize(inputs[:-1], budget=32, joint_tiny=True)["paired_scores"] == []


@pytest.mark.parametrize("corruption", ["mode", "microbatch", "rate", "provenance", "shared", "policy"])
def test_joint_tiny_rejects_unmatched_arms(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr("kindle._vector_audit.audit", lambda _: dict(accounting_valid=True))
    frozen, joint = joint_fixture(1009, "frozen"), joint_fixture(1009, "joint")
    if corruption == "provenance":
        joint[0]["model_provenance"]["meganeura_revision"] = "changed"
    else:
        field, value = dict(mode=("video_encoder", "frozen"), microbatch=("world_microbatch_size", 8),
                            rate=("learning_rate", 3e-4), shared=("horizon", 42),
                            policy=("actor_critic_gradient", True))[corruption]
        joint[0]["config"][field] = value
    inputs = [("frozen_tiny", write(tmp_path, frozen, "frozen.jsonl")),
              ("joint_tiny", write(tmp_path, joint, "joint.jsonl"))]
    with pytest.raises(ValueError):
        summarize(inputs, budget=32, joint_tiny=True)


def test_policy_tiny_reuses_task_controls_but_requires_the_direct_gradient_flag(tmp_path, monkeypatch):
    monkeypatch.setattr("kindle._vector_audit.audit", lambda _: dict(accounting_valid=True))
    inputs = []
    for seed in (1009, 2017, 3019):
        for enabled in (False, True):
            method = "policy_tiny" if enabled else "joint_tiny"
            rows = joint_fixture(seed, "joint", reward=1 + enabled)
            # Earlier task-only controls have no field; absent is default-off.
            if enabled:
                rows[0]["config"]["actor_critic_gradient"] = True
            inputs.append((method, write(tmp_path, rows, f"{method}-{seed}.jsonl")))
    result = summarize(inputs, budget=32, policy_tiny=True)
    assert result["status"] == "learning_comparison_complete"
    assert result["paired_scores"][0]["difference"] == dict(mean=3., ci95=[3., 3.], seeds=[3.]*3)
    assert "Direct-policy Tiny · online" in plot_svg(result)
    assert "initial posterior states only" in markdown(result, "data.json")
    assert summarize(inputs[:-1], budget=32, policy_tiny=True)["paired_scores"] == []
    for field, value in (("actor_critic_gradient", False), ("video_encoder", "frozen")):
        broken = copy.deepcopy(rows)
        broken[0]["config"][field] = value
        with pytest.raises(ValueError, match="label differs"):
            summarize([("policy_tiny", write(tmp_path, broken, "broken.jsonl"))], budget=32, policy_tiny=True)


def test_small_representation_reuses_curves_without_claiming_replication(tmp_path):
    inputs = []
    for seed in (1009, 2017, 3019):
        for method in ("learned_cnn", *TINY_CHECKPOINTS):
            rows = fixture(seed, replication=True) if method == "learned_cnn" else tiny_fixture(seed, method)
            inputs.append((method, write(tmp_path, rows, f"{method}-{seed}.jsonl")))
    result = summarize(inputs, budget=32, small_representation=True)
    assert result["status"] == "learning_comparison_complete" and not result["phase2_complete"]
    assert result["num_envs"] == 8 and len(result["results"]) == 3
    assert all(r["aggregate"] for r in result["results"])
    assert "Seaquest · 9/9 runs" in plot_svg(result)
    assert "250k" in markdown(result, "data.json") and "whole packages" in " ".join(result["limits"])
    partial = summarize(inputs[:-1], budget=32, small_representation=True)
    assert partial["status"] == "partial_learning_comparison"
    assert next(r for r in partial["results"] if r["method"] == "initial_tiny")["aggregate"] is None
    with pytest.raises(ValueError, match="choose one learning comparison"):
        summarize(inputs, budget=32, replication=True, small_representation=True)


@pytest.mark.parametrize("corruption", ["checkpoint", "kind", "loss", "pixels", "schedule"])
def test_small_representation_rejects_frontend_and_recipe_mislabeling(tmp_path, corruption):
    rows = tiny_fixture(1009, "pretrained_tiny")
    if corruption == "checkpoint":
        rows[0]["model_provenance"]["perception"]["checkpoint_sha256"] = TINY_CHECKPOINTS["initial_tiny"]
    elif corruption == "kind":
        rows[0]["model_provenance"]["perception"]["kind"] = "levjepa"
    elif corruption == "loss":
        rows[0]["config"]["loss_scales"]["reconstruction"] = 1.
    elif corruption == "pixels":
        rows[0]["observation_size"] = "64"
    else:
        rows[0]["config"]["batch_length"] = 64
    with pytest.raises(ValueError):
        summarize([("pretrained_tiny", write(tmp_path, rows))], budget=32, small_representation=True)


def test_small_representation_matches_optimizer_settings_across_frontends(tmp_path):
    rgb = fixture(replication=True)
    tiny = tiny_fixture(1009, "pretrained_tiny")
    rgb[0]["config"]["learning_rate"] = 4e-5
    tiny[0]["config"]["learning_rate"] = 3e-4
    inputs = [("learned_cnn", write(tmp_path, rgb, "rgb.jsonl")),
              ("pretrained_tiny", write(tmp_path, tiny, "tiny.jsonl"))]
    with pytest.raises(ValueError, match="shared learner"):
        summarize(inputs, budget=32, small_representation=True)


def test_small_comparison_pairs_seeds_before_bootstrap_and_omits_incomplete_pairs(tmp_path):
    inputs = []
    for seed, baseline, gain in ((1009, 10., 1.), (2017, 100., 2.), (3019, 1000., 3.)):
        for method, rows in (
            ("learned_cnn", fixture(seed, replication=True, reward=baseline)),
            ("pretrained_tiny", tiny_fixture(seed, "pretrained_tiny", reward=baseline + gain)),
        ):
            inputs.append((method, write(tmp_path, rows, f"{method}-{seed}.jsonl")))
    result = summarize(list(reversed(inputs)), budget=32, small_representation=True)
    assert result["paired_scores"] == [dict(candidate="pretrained_tiny", control="learned_cnn",
        game="Seaquest", seeds=(1009, 2017, 3019), difference=dict(mean=6., ci95=[3., 9.], seeds=[3., 6., 9.]))]
    assert "6.000 [3.000, 9.000]" in markdown(result, "data.json")
    assert summarize(inputs[:-1], budget=32, small_representation=True)["paired_scores"] == []
