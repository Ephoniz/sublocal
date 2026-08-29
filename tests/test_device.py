from __future__ import annotations

import os
from pathlib import Path

import pytest

from sublocal.cli import main
from sublocal.device import (
    CudaUnavailableError,
    resolve_device,
    unhide_cuda_env,
)


def test_unhide_drops_minus_one(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert unhide_cuda_env() == "-1"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_unhide_drops_empty(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert unhide_cuda_env() == ""
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_unhide_keeps_real_list(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert unhide_cuda_env() is None
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_auto_uses_cuda_when_count_positive(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 1)
    monkeypatch.setattr(
        "sublocal.device.cuda_device_name",
        lambda index=0: "NVIDIA GeForce RTX 4070 Ti",
    )
    assert resolve_device("auto") == "cuda"
    err = capsys.readouterr().err
    assert "Using NVIDIA GeForce RTX 4070 Ti (cuda:0)" in err
    assert "Using CPU" not in err


def test_auto_cpu_explains_count_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 0)
    assert resolve_device("auto") == "cpu"
    err = capsys.readouterr().err
    assert "Using CPU" in err
    assert "get_cuda_device_count()=0" in err


def test_cuda_count_zero_raises(monkeypatch) -> None:
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 0)
    with pytest.raises(CudaUnavailableError, match="get_cuda_device_count\\(\\)=0"):
        resolve_device("cuda")


def test_minus_one_ignored_for_auto(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    seen: dict[str, str | None] = {}

    def fake_count() -> int:
        seen["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return 1

    monkeypatch.setattr("sublocal.device.cuda_device_count", fake_count)
    monkeypatch.setattr(
        "sublocal.device.cuda_device_name", lambda index=0: "NVIDIA GeForce RTX 4070 Ti"
    )
    assert resolve_device("auto") == "cuda"
    assert seen["env"] is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "Using NVIDIA GeForce RTX 4070 Ti (cuda:0)" in capsys.readouterr().err


def test_minus_one_ignored_for_cuda(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    seen: dict[str, str | None] = {}

    def fake_count() -> int:
        seen["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return 1

    monkeypatch.setattr("sublocal.device.cuda_device_count", fake_count)
    monkeypatch.setattr("sublocal.device.cuda_device_name", lambda index=0: "Fake GPU")
    assert resolve_device("cuda") == "cuda"
    assert seen["env"] is None
    assert "Using Fake GPU (cuda:0)" in capsys.readouterr().err


def test_cpu_says_so(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert resolve_device("cpu") == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert capsys.readouterr().err.strip() == "Using CPU"


def test_cli_cuda_count_zero_exits_1(
    sample_srt: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 0)
    rc = main(
        [
            "translate",
            str(sample_srt),
            "--to",
            "en",
            "--from",
            "es",
            "--device",
            "cuda",
            "--backend",
            "nllb",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "get_cuda_device_count()=0" in captured.err
    assert "Translating" not in captured.err
