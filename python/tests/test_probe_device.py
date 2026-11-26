from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from fit_atari_probes import checked_memory


@pytest.fixture
def model(monkeypatch):
    monkeypatch.delenv("KINDLE_EXPECT_DEVICE_NAME", raising=False)
    monkeypatch.delenv("KINDLE_EXPECT_DRIVER_INFO", raising=False)
    return SimpleNamespace(
        gpu_device=dict(device_name="NVIDIA GeForce RTX 5080", driver_info="615.71.09"),
        gpu_memory_budget=dict(budget_bytes=4 << 30, usage_bytes=2 << 30),
    )


def test_driver_is_not_pinned_to_a_historical_host(model):
    assert checked_memory(model) == model.gpu_memory_budget


def test_declared_driver_is_enforced(model, monkeypatch):
    monkeypatch.setenv("KINDLE_EXPECT_DRIVER_INFO", "615.71.09")
    checked_memory(model)
    model.gpu_device["driver_info"] = "different"
    with pytest.raises(RuntimeError, match="unexpected probe GPU"):
        checked_memory(model)


def test_declared_device_is_enforced(model, monkeypatch):
    monkeypatch.setenv("KINDLE_EXPECT_DEVICE_NAME", "Test device")
    with pytest.raises(RuntimeError, match="unexpected probe GPU"):
        checked_memory(model)
    model.gpu_device["device_name"] = "Test device"
    checked_memory(model)


@pytest.mark.parametrize("memory", [None, dict(budget_bytes=4 << 30, usage_bytes=(2 << 30) + 1)])
def test_headroom_bound_is_unchanged(model, memory):
    model.gpu_memory_budget = memory
    with pytest.raises(RuntimeError, match="less than 2GiB"):
        checked_memory(model)
