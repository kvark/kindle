import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kindle._atari_scores import (
    UPSTREAM_D3_REFERENCE,
    AtariScoreError,
    compare_runtime_scores,
    load_segments,
    load_upstream_d3_segments,
    summarize_scores,
)


def test_score_module_does_not_load_native_extension() -> None:
    python_root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(python_root)}
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import kindle._atari_scores; "
                "assert 'kindle._native' not in sys.modules"
            ),
        ],
        check=True,
        cwd=python_root.parent,
        env=environment,
    )


def write_run(
    path,
    *,
    seed,
    start,
    end,
    episodes,
    protocol="published",
    full_action_space=True,
    config=None,
    random_action_steps=0,
):
    records = [
        {
            "event": "run_start",
            "environment": "ALE/Pong-v5",
            "seed": seed,
            "mode": "train",
            "atari_protocol": protocol,
            "action_repeat": 4,
            "full_action_space": full_action_space,
            "noop_max": 0,
            "max_episode_frames": 100_000,
            "score_window_frames": [350_000, 400_000],
            "starting_environment_step": start // 4,
            "config": config,
            "random_action_steps": random_action_steps,
        },
        *[
            {
                "event": "episode",
                "environment_frames": frame,
                "return": score,
            }
            for frame, score in episodes
        ],
        {
            "event": "run_end",
            "environment": "ALE/Pong-v5",
            "seed": seed,
            "environment_frames": end,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def write_upstream_run(
    logdir, *, seed=0, episodes=(), complete=True, manifest_update=None
):
    logdir.mkdir()
    manifest = {}
    for path, value in UPSTREAM_D3_REFERENCE.items():
        target = manifest
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = value
    manifest["experiment"]["seed"] = seed
    if manifest_update:
        manifest_update(manifest)
    (logdir / "reference-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (logdir / "scores.jsonl").write_text(
        "".join(
            json.dumps({"step": step, "episode/score": score}) + "\n"
            for step, score in episodes
        ),
        encoding="utf-8",
    )
    if complete:
        (logdir / "RUN_COMPLETE").write_text("", encoding="utf-8")


def test_scores_average_episodes_per_seed_before_seeds(tmp_path) -> None:
    seed0 = tmp_path / "seed0.jsonl"
    seed1 = tmp_path / "seed1.jsonl"
    write_run(seed0, seed=0, start=0, end=400_000, episodes=[(360_000, 10)])
    write_run(
        seed1,
        seed=1,
        start=0,
        end=400_000,
        episodes=[(350_000, 0), (375_000, 0), (400_000, 0)],
    )

    summary = summarize_scores(load_segments([seed0, seed1]))
    assert summary["require_uninterrupted"] is False
    pong = summary["environments"]["ALE/Pong-v5"]
    assert pong["target_runtime"] == "kindle-ale_py-0.12.1"
    assert pong["seed_mean"] == 5.0
    assert [seed["completed_episodes"] for seed in pong["seed_results"]] == [1, 3]
    assert pong["targets"]["dreamerv3_12m_reported"] == {
        "score": -10.0,
        "delta": 15.0,
        "met": True,
    }


def test_dreamerv3_12m_target_boundary_is_inclusive(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}.jsonl"
        write_run(
            path,
            seed=seed,
            start=0,
            end=400_000,
            episodes=[(380_000, -10.0)],
        )
        paths.append(path)

    summary = summarize_scores(
        load_segments(paths),
        minimum_seeds=3,
        require_uninterrupted=True,
    )
    target = summary["environments"]["ALE/Pong-v5"]["targets"][
        "dreamerv3_12m_reported"
    ]
    assert target == {"score": -10.0, "delta": 0.0, "met": True}


def test_dreamerv3_12m_target_rejects_score_just_below_boundary(tmp_path) -> None:
    paths = []
    for seed, score in enumerate((-10.0, -10.0, -10.003)):
        path = tmp_path / f"seed{seed}.jsonl"
        write_run(
            path,
            seed=seed,
            start=0,
            end=400_000,
            episodes=[(380_000, score)],
        )
        paths.append(path)

    summary = summarize_scores(
        load_segments(paths),
        minimum_seeds=3,
        require_uninterrupted=True,
    )
    target = summary["environments"]["ALE/Pong-v5"]["targets"][
        "dreamerv3_12m_reported"
    ]
    assert target["score"] == -10.0
    assert target["delta"] == pytest.approx(-0.001)
    assert target["met"] is False


def test_scores_enforce_minimum_independent_seed_count(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}.jsonl"
        write_run(
            path,
            seed=seed,
            start=0,
            end=400_000,
            episodes=[(380_000, -10 + seed)],
        )
        paths.append(path)

    with pytest.raises(
        AtariScoreError,
        match="requires at least 3 independent seeds, found 2",
    ):
        summarize_scores(load_segments(paths[:2]), minimum_seeds=3)

    summary = summarize_scores(load_segments(paths), minimum_seeds=3)
    assert summary["minimum_seed_count"] == 3
    assert summary["environments"]["ALE/Pong-v5"]["seed_count"] == 3


def test_scores_require_one_experiment_identity_across_seeds(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"seed{seed}.jsonl"
        write_run(
            path,
            seed=seed,
            start=0,
            end=400_000,
            episodes=[(380_000, -10.0)],
            config={"model_size": "size12_m", "seed": seed},
        )
        paths.append(path)

    summary = summarize_scores(load_segments(paths), minimum_seeds=3)
    identity = summary["environments"]["ALE/Pong-v5"]["experiment_identity"]
    assert identity["config_without_seed"] == {"model_size": "size12_m"}

    write_run(
        paths[2],
        seed=2,
        start=0,
        end=400_000,
        episodes=[(380_000, -10.0)],
        config={"model_size": "size1_m", "seed": 2},
    )
    with pytest.raises(AtariScoreError, match="different experiment identities"):
        summarize_scores(load_segments(paths), minimum_seeds=3)


def test_scores_reject_config_seed_mismatch(tmp_path) -> None:
    path = tmp_path / "seed0.jsonl"
    write_run(
        path,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(380_000, -10.0)],
        config={"model_size": "size12_m", "seed": 1},
    )
    with pytest.raises(AtariScoreError, match="run seed 0 does not match config seed 1"):
        load_segments([path])


def test_multiple_segments_do_not_inflate_independent_seed_count(tmp_path) -> None:
    first = tmp_path / "seed0-first.jsonl"
    second = tmp_path / "seed0-second.jsonl"
    write_run(first, seed=0, start=0, end=375_000, episodes=[(360_000, -10)])
    write_run(
        second,
        seed=0,
        start=375_000,
        end=400_000,
        episodes=[(390_000, -10)],
    )

    with pytest.raises(
        AtariScoreError,
        match="requires at least 3 independent seeds, found 1",
    ):
        summarize_scores(load_segments([first, second]), minimum_seeds=3)


@pytest.mark.parametrize("minimum_seeds", [True, 0, -1, 1.5])
def test_scores_reject_invalid_minimum_seed_count(minimum_seeds) -> None:
    with pytest.raises(AtariScoreError, match="minimum_seeds"):
        summarize_scores([], minimum_seeds=minimum_seeds)


def test_resumed_segments_can_jointly_cover_score_window(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_run(first, seed=0, start=0, end=375_000, episodes=[(360_000, -10)])
    write_run(
        second,
        seed=0,
        start=375_000,
        end=400_000,
        episodes=[(390_000, -4)],
    )

    summary = summarize_scores(load_segments([first, second]))
    pong = summary["environments"]["ALE/Pong-v5"]
    assert pong["seed_results"] == [
        {"seed": 0, "score": -7.0, "completed_episodes": 2, "segment_count": 2}
    ]


def test_appended_checkpoint_recovery_discards_superseded_tail(tmp_path) -> None:
    path = tmp_path / "recovered.jsonl"
    common = {
        "environment": "ALE/Pong-v5",
        "seed": 0,
        "mode": "train",
        "atari_protocol": "published",
        "action_repeat": 4,
        "full_action_space": True,
        "noop_max": 0,
        "max_episode_frames": 100_000,
        "score_window_frames": [350_000, 400_000],
        "actions": 18,
        "action_meanings": ["NOOP", "FIRE"],
        "dino_model_id": "test-dino",
        "dino_checkpoint_revision": "revision",
        "dino_checkpoint_sha256": "sha256",
        "trainable_parameters": {"total": 1},
        "config": {"batch_length": 64},
    }
    records = [
        {"event": "run_start", "starting_environment_step": 0, **common},
        {"event": "episode", "environment_frames": 352_000, "return": -18},
        {
            "event": "checkpoint",
            "environment_step": 90_000,
            "learner_step": 22_229,
        },
        {"event": "episode", "environment_frames": 368_000, "return": 21},
        {
            "event": "run_start",
            "starting_environment_step": 90_000,
            "starting_learner_step": 22_229,
            "output_appended": True,
            **common,
        },
        {"event": "episode", "environment_frames": 380_000, "return": -10},
        {
            "event": "run_end",
            "environment": "ALE/Pong-v5",
            "seed": 0,
            "environment_frames": 400_000,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_scores(load_segments([path]))
    assert summary["environments"]["ALE/Pong-v5"]["seed_results"] == [
        {
            "seed": 0,
            "score": -14.0,
            "completed_episodes": 2,
            "segment_count": 2,
            "checkpoint_recoveries": 1,
        }
    ]

    with pytest.raises(
        AtariScoreError,
        match="uninterrupted scoring rejects 1 checkpoint recovery",
    ):
        summarize_scores(
            load_segments([path]),
            require_uninterrupted=True,
        )

    python_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(python_root / "examples" / "summarize_atari_scores.py"),
            str(path),
            "--require-uninterrupted-kindle",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(python_root)},
    )
    assert result.returncode == 2
    assert "uninterrupted scoring rejects 1 checkpoint recovery" in result.stderr


def test_appended_checkpoint_recovery_requires_matching_checkpoint(tmp_path) -> None:
    path = tmp_path / "unproven-recovery.jsonl"
    common = {
        "environment": "ALE/Pong-v5",
        "seed": 0,
        "mode": "train",
        "atari_protocol": "published",
        "action_repeat": 4,
        "full_action_space": True,
        "noop_max": 0,
        "max_episode_frames": 100_000,
        "score_window_frames": [350_000, 400_000],
    }
    records = [
        {"event": "run_start", "starting_environment_step": 0, **common},
        {
            "event": "run_start",
            "starting_environment_step": 90_000,
            "starting_learner_step": 22_229,
            "output_appended": True,
            **common,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(AtariScoreError, match="no matching checkpoint event"):
        load_segments([path])


def test_incomplete_window_is_rejected(tmp_path) -> None:
    path = tmp_path / "incomplete.jsonl"
    write_run(path, seed=0, start=0, end=390_000, episodes=[(380_000, -10)])

    with pytest.raises(AtariScoreError, match="do not cover the full"):
        summarize_scores(load_segments([path]))


def test_published_profile_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "current.jsonl"
    write_run(
        path,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(380_000, -10)],
        protocol="current",
    )

    with pytest.raises(AtariScoreError, match="expected 'published' protocol"):
        summarize_scores(load_segments([path]))


def test_published_minimal_uses_only_its_matched_random_target(tmp_path) -> None:
    path = tmp_path / "minimal.jsonl"
    write_run(
        path,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(380_000, -20)],
        protocol="published-minimal",
        full_action_space=False,
    )

    summary = summarize_scores(
        load_segments([path]), expected_protocol="published-minimal"
    )
    targets = summary["environments"]["ALE/Pong-v5"]["targets"]
    assert targets == {
        "matched_random_3_seed": {
            "score": -20.446031746031746,
            "delta": 0.4460317460317462,
            "met": True,
        }
    }


def test_published_minimal_action_metadata_is_enforced(tmp_path) -> None:
    path = tmp_path / "minimal-all-actions.jsonl"
    write_run(
        path,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(380_000, -20)],
        protocol="published-minimal",
        full_action_space=True,
    )

    with pytest.raises(AtariScoreError, match="requires full_action_space=False"):
        summarize_scores(
            load_segments([path]), expected_protocol="published-minimal"
        )


def test_completed_upstream_d3_run_uses_the_same_score_window(tmp_path) -> None:
    logdir = tmp_path / "upstream"
    write_upstream_run(
        logdir,
        episodes=[(340_000, -21), (360_000, -6), (380_000, -2)],
    )

    summary = summarize_scores(load_upstream_d3_segments([logdir]))
    pong = summary["environments"]["ALE/Pong-v5"]
    assert pong["target_runtime"] == "upstream-d3-ale_py-0.9.0"
    assert pong["seed_results"] == [
        {"seed": 0, "score": -4.0, "completed_episodes": 2, "segment_count": 1}
    ]
    assert pong["targets"]["dreamerv3_200m_published"] == {
        "score": -4.537,
        "delta": pytest.approx(0.537),
        "met": True,
    }
    assert pong["targets"]["matched_random_3_seed"] == {
        "score": pytest.approx(-20.373626373626372),
        "delta": pytest.approx(16.373626373626372),
        "met": True,
    }


def test_runtime_comparison_removes_each_ale_random_floor(tmp_path) -> None:
    kindle_log = tmp_path / "kindle.jsonl"
    upstream_logdir = tmp_path / "upstream"
    write_run(
        kindle_log,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(360_000, -12), (380_000, -8)],
    )
    write_upstream_run(
        upstream_logdir,
        episodes=[(360_000, -10), (380_000, -8)],
    )

    kindle = summarize_scores(load_segments([kindle_log]))
    upstream = summarize_scores(load_upstream_d3_segments([upstream_logdir]))
    comparison = compare_runtime_scores(kindle, upstream)
    assert comparison["minimum_seed_counts"] == {"kindle": 1, "upstream_d3": 1}
    assert comparison["require_uninterrupted"] == {
        "kindle": False,
        "upstream_d3": False,
    }
    pong = comparison["comparisons"]["ALE/Pong-v5"]
    assert pong == {
        "kindle_score": -10.0,
        "upstream_d3_score": -9.0,
        "raw_score_delta": -1.0,
        "kindle_random_score": pytest.approx(-20.622710622710624),
        "upstream_d3_random_score": pytest.approx(-20.373626373626372),
        "kindle_gain_over_random": pytest.approx(10.622710622710624),
        "upstream_d3_gain_over_random": pytest.approx(11.373626373626372),
        "random_adjusted_delta": pytest.approx(-0.7509157509157489),
    }


def test_runtime_comparison_rejects_reversed_inputs(tmp_path) -> None:
    kindle_log = tmp_path / "kindle.jsonl"
    upstream_logdir = tmp_path / "upstream"
    write_run(
        kindle_log,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(380_000, -10)],
    )
    write_upstream_run(upstream_logdir, episodes=[(380_000, -9)])
    kindle = summarize_scores(load_segments([kindle_log]))
    upstream = summarize_scores(load_upstream_d3_segments([upstream_logdir]))

    with pytest.raises(AtariScoreError, match="first summary is not the Kindle"):
        compare_runtime_scores(upstream, kindle)


def test_score_cli_compares_kindle_and_upstream_runs(tmp_path) -> None:
    kindle_log = tmp_path / "kindle.jsonl"
    upstream_logdir = tmp_path / "upstream"
    write_run(
        kindle_log,
        seed=0,
        start=0,
        end=400_000,
        episodes=[(380_000, -10)],
    )
    write_upstream_run(upstream_logdir, episodes=[(380_000, -9)])
    python_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(python_root / "examples" / "summarize_atari_scores.py"),
            str(kindle_log),
            "--upstream-d3-logdir",
            str(upstream_logdir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(python_root)},
    )

    comparison = json.loads(result.stdout)
    pong = comparison["comparisons"]["ALE/Pong-v5"]
    assert pong["raw_score_delta"] == -1.0
    assert pong["random_adjusted_delta"] == pytest.approx(-0.7509157509157489)


def test_score_summary_rejects_mixed_target_runtimes(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_run(first, seed=0, start=0, end=400_000, episodes=[(380_000, -20)])
    write_run(second, seed=1, start=0, end=400_000, episodes=[(380_000, -20)])
    segments = load_segments([first, second])
    segments[1]["target_runtime"] = "kindle-ale_py-0.13.0"

    with pytest.raises(AtariScoreError, match="cannot mix target runtimes"):
        summarize_scores(segments)


def test_upstream_d3_run_requires_completion_evidence(tmp_path) -> None:
    logdir = tmp_path / "upstream"
    write_upstream_run(logdir, episodes=[(380_000, -4)], complete=False)

    with pytest.raises(AtariScoreError, match="no RUN_COMPLETE evidence"):
        load_upstream_d3_segments([logdir])


def test_upstream_d3_run_rejects_provenance_mismatch(tmp_path) -> None:
    logdir = tmp_path / "upstream"
    write_upstream_run(
        logdir,
        episodes=[(380_000, -4)],
        manifest_update=lambda manifest: manifest["runtime"].update(
            {"ale_py": "0.12.1"}
        ),
    )

    with pytest.raises(AtariScoreError, match="runtime.ale_py='0.9.0'"):
        load_upstream_d3_segments([logdir])


def test_upstream_d3_run_rejects_duplicate_episode_steps(tmp_path) -> None:
    logdir = tmp_path / "upstream"
    write_upstream_run(logdir, episodes=[(380_000, -4), (380_000, -3)])

    with pytest.raises(AtariScoreError, match="duplicate episode step 380000"):
        load_upstream_d3_segments([logdir])


def test_upstream_d3_run_rejects_non_finite_score(tmp_path) -> None:
    logdir = tmp_path / "upstream"
    write_upstream_run(logdir, episodes=[(380_000, float("nan"))])

    with pytest.raises(AtariScoreError, match="episode/score must be finite"):
        load_upstream_d3_segments([logdir])
