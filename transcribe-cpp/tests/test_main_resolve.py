"""Bootstrap model resolution: custom_model failures stop the add-on."""

from pathlib import Path

import pytest

from wyoming_transcribe_cpp import __main__ as main_mod
from wyoming_transcribe_cpp import convert
from wyoming_transcribe_cpp.__main__ import _parse_args, _resolve_model
from wyoming_transcribe_cpp.const import EXIT_BOOTSTRAP
from wyoming_transcribe_cpp.convert import ConversionFailed
from wyoming_transcribe_cpp.model_options import parse_config as parse_model_options
from wyoming_transcribe_cpp.models import DEFAULT_MODEL, REGISTRY


class TestResolveModel:
    def test_catalog_model_without_custom(self, monkeypatch):
        args = _parse_args([])
        monkeypatch.setattr(
            main_mod, "ensure_gguf",
            lambda m, q, d, t: Path(f"/data/models/{m}.gguf"),
        )
        gguf, name, repo = _resolve_model(args, None)
        assert name == DEFAULT_MODEL
        assert repo == REGISTRY[DEFAULT_MODEL].repo

    def test_custom_model_served_when_conversion_succeeds(self, monkeypatch):
        args = _parse_args(["--custom-model", "me/ft"])
        monkeypatch.setattr(
            convert, "request_conversion",
            lambda *a, **kw: Path("/data/models/custom/me__ft-Q4_K_M.gguf"),
        )
        gguf, name, repo = _resolve_model(args, None)
        assert name == "me/ft"
        assert gguf.name == "me__ft-Q4_K_M.gguf"

    def test_conversion_failure_logs_and_stops(self, monkeypatch, caplog):
        # Policy: a broken custom_model is a configuration error — the
        # add-on stops (s6 finish halts the container) rather than
        # silently serving a model the user did not select.
        args = _parse_args(["--custom-model", "me/ft"])

        def boom(*a, **kw):
            raise ConversionFailed("converter died with SIGKILL")

        monkeypatch.setattr(convert, "request_conversion", boom)
        with caplog.at_level("ERROR"), pytest.raises(ConversionFailed):
            _resolve_model(args, None)
        assert any("stopping" in r.message for r in caplog.records)

    def test_worker_socket_outage_also_stops(self, monkeypatch):
        args = _parse_args(["--custom-model", "me/ft"])

        def boom(*a, **kw):
            raise OSError("connect: no such file or directory")

        monkeypatch.setattr(convert, "request_conversion", boom)
        with pytest.raises(OSError):
            _resolve_model(args, None)


class TestExitCodePolicy:
    """The exit code picks the recovery policy the s6 finish handler applies.

    Bootstrap failures are deterministic — the finish handler turns
    EXIT_BOOTSTRAP into a container exit of 0 so the Supervisor watchdog
    leaves the app stopped instead of restart-looping a broken config.
    A crash after startup exits 1 and is worth restarting.
    """

    def test_bootstrap_failure_exits_with_exit_bootstrap(self, monkeypatch):
        monkeypatch.setattr(main_mod, "_serving", False)

        async def boom():
            raise ConversionFailed("converter died")

        monkeypatch.setattr(main_mod, "main", boom)
        with pytest.raises(SystemExit) as exc:
            main_mod.run()
        assert exc.value.code == EXIT_BOOTSTRAP

    def test_crash_after_startup_exits_one(self, monkeypatch):
        monkeypatch.setattr(main_mod, "_serving", True)

        async def boom():
            raise RuntimeError("segfault-adjacent runtime blowup")

        monkeypatch.setattr(main_mod, "main", boom)
        with pytest.raises(SystemExit) as exc:
            main_mod.run()
        assert exc.value.code == 1

    def test_finish_handler_agrees_with_const(self):
        finish = (
            Path(__file__).parent.parent
            / "rootfs/etc/s6-overlay/s6-rc.d/transcribe-cpp/finish"
        ).read_text()
        assert f"readonly exit_code_bootstrap={EXIT_BOOTSTRAP}" in finish


class TestModelOptionsFlag:
    def test_default_is_empty_json_array(self):
        args = _parse_args([])
        assert args.model_options_json == "[]"
        assert parse_model_options(args.model_options_json) == {}

    def test_round_trips_through_parse_config(self):
        raw = '[{"name": "att_context_right", "value": "6"}]'
        args = _parse_args(["--model-options-json", raw])
        assert parse_model_options(args.model_options_json) == {"att_context_right": "6"}
