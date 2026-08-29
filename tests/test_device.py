from __future__ import annotations

import os
from pathlib import Path

import pytest

from sublocal.cli import main
from sublocal.device import (
    CudaUnavailableError,
    _cuda_missing_message,
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
    first = next((ln for ln in err.splitlines() if ln.strip()), "")
    assert not first.startswith("Using CPU")
    assert "get_cuda_device_count()=0" in err
    assert "CUDA_PATH" in err or "cublas64_12" in err


def test_cuda_count_zero_raises(monkeypatch) -> None:
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 0)
    with pytest.raises(CudaUnavailableError) as excinfo:
        resolve_device("cuda")
    msg = str(excinfo.value)
    assert "get_cuda_device_count()=0" in msg
    assert "CUDA_PATH" in msg or "cublas64_12" in msg
    assert "Using CPU" not in msg


def test_missing_message_cuda_path_v13(monkeypatch) -> None:
    monkeypatch.setenv(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    msg = _cuda_missing_message(requested="auto", hidden=None)
    assert "CUDA_PATH=" in msg
    assert "v13.1" in msg
    assert "12.x bin on PATH" in msg
    assert "cublas64_12.dll" in msg


def test_missing_message_missing_cublas(monkeypatch, tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "CUDA" / "v12.6"))
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    msg = _cuda_missing_message(requested="auto", hidden=None)
    assert "cublas64_12.dll not found on PATH or CUDA_PATH\\bin" in msg
    assert "cudart64_12.dll not found on PATH or CUDA_PATH\\bin" in msg


def test_missing_message_empty_visible_devices(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr("sublocal.device.cuda_device_count", lambda: 0)
    assert resolve_device("auto") == "cpu"
    err = capsys.readouterr().err
    assert "leftover empty CUDA_VISIBLE_DEVICES" in err
    assert "get_cuda_device_count()=0" in err
    first = next((ln for ln in err.splitlines() if ln.strip()), "")
    assert not first.startswith("Using CPU")


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
    assert "CUDA_PATH" in captured.err or "cublas64_12" in captured.err
    assert "Translating" not in captured.err
    assert "Using CPU" not in captured.err
