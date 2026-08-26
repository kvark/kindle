import json

import pytest

from kindle._atari_scores import (
    UPSTREAM_D3_REFERENCE,
    AtariScoreError,
    load_segments,
    load_upstream_d3_segments,
    summarize_scores,
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
    pong = summary["environments"]["ALE/Pong-v5"]
    assert pong["seed_mean"] == 5.0
    assert [seed["completed_episodes"] for seed in pong["seed_results"]] == [1, 3]
    assert pong["targets"]["dreamerv3_12m_reported"] == {
        "score": -10.0,
        "delta": 15.0,
        "met": True,
    }


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
    assert pong["seed_results"] == [
        {"seed": 0, "score": -4.0, "completed_episodes": 2, "segment_count": 1}
    ]
    assert pong["targets"]["dreamerv3_200m_published"] == {
        "score": -4.537,
        "delta": pytest.approx(0.537),
        "met": True,
    }


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
