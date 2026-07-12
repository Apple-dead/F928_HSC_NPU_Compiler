#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
BASIC_ROOT = TEST_ROOT / "basic"
ERROR_DIR = TEST_ROOT / "error"
PYTHON_CONFIG = PROJECT_ROOT / "python" / "npu_config.py"


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def command_text(self) -> str:
        return " ".join(self.args)


@dataclass
class BasicTestResult:
    name: str
    passed: bool
    message: str
    log_path: Path | None = None


class BasicTestFailure(AssertionError):
    pass


def build_command(os_name: str, *extra: str) -> list[str]:
    if os_name == "windows":
        return ["cmd", "/c", "build.bat", *extra]
    return ["bash", "./build.sh", *extra]


def run_command(args: list[str]) -> CommandResult:
    completed = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(
        args=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def format_command_result(result: CommandResult) -> str:
    return (
        f"command: {result.command_text}\n"
        f"returncode: {result.returncode}\n\n"
        f"[stdout]\n{result.stdout}\n\n"
        f"[stderr]\n{result.stderr}"
    )


def write_log(test_name: str, title: str, sections: list[tuple[str, str]]) -> Path:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ERROR_DIR / f"{test_name}_{timestamp}.log.txt"
    lines = [f"# {title}", "", f"test = {test_name}", ""]
    for section_title, body in sections:
        lines.append(f"## {section_title}")
        lines.append(body.rstrip() if body else "<empty>")
        lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"unsupported config patch value: {value!r}")


def patch_config_text(text: str, patch: dict[str, Any]) -> str:
    result = text
    for key, value in patch.items():
        replacement = f"{key} = {python_literal(value)}"
        pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
        result, count = pattern.subn(replacement, result)
        if count == 0:
            if not result.endswith("\n"):
                result += "\n"
            result += replacement + "\n"
    return result


class BasicContext:
    def __init__(self, *, os_name: str, base_config: bytes) -> None:
        self.os_name = os_name
        self.base_config = base_config
        self.commands: list[CommandResult] = []

    def restore_config(self) -> None:
        PYTHON_CONFIG.write_bytes(self.base_config)

    def apply_config_patch(self, patch: dict[str, Any]) -> None:
        text = self.base_config.decode("utf-8")
        PYTHON_CONFIG.write_text(patch_config_text(text, patch), encoding="utf-8", newline="\n")

    def clean(self) -> None:
        result = run_command(build_command(self.os_name, "clean"))
        self.commands.append(result)
        if result.returncode != 0:
            raise BasicTestFailure("build clean failed")

    def build_with_config_patch(self, patch: dict[str, Any] | None = None) -> None:
        self.clean()
        self.restore_config()
        if patch:
            self.apply_config_patch(patch)
        result = run_command(build_command(self.os_name))
        self.commands.append(result)
        if result.returncode != 0:
            raise BasicTestFailure("build failed")

    def read_json(self, relative_path: str) -> Any:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise BasicTestFailure(f"missing JSON file: {relative_path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def read_text(self, relative_path: str) -> str:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise BasicTestFailure(f"missing text file: {relative_path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def require_file(self, relative_path: str) -> Path:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise BasicTestFailure(f"missing required file: {relative_path}")
        return path

    def require_absent(self, relative_path: str) -> None:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            raise BasicTestFailure(f"path should have been removed by clean: {relative_path}")

    def sha256(self, relative_path: str) -> str:
        path = self.require_file(relative_path)
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def command_log(self) -> str:
        return "\n\n".join(format_command_result(result) for result in self.commands)


def discover_basic_tests() -> list[Path]:
    if not BASIC_ROOT.is_dir():
        return []
    return sorted(BASIC_ROOT.glob("test_*.py"))


def load_run_function(path: Path) -> Callable[[BasicContext], None]:
    spec = importlib.util.spec_from_file_location(f"basic_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise BasicTestFailure(f"could not load basic test: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise BasicTestFailure(f"{path.name} must expose run(ctx)")
    return run


def run_basic_tests(os_name: str, *, base_config: bytes | None = None) -> list[BasicTestResult]:
    tests = discover_basic_tests()
    if not tests:
        return []

    config = base_config if base_config is not None else PYTHON_CONFIG.read_bytes()
    results: list[BasicTestResult] = []
    for path in tests:
        test_name = f"basic/{path.stem}"
        ctx = BasicContext(os_name=os_name, base_config=config)
        try:
            ctx.restore_config()
            run = load_run_function(path)
            run(ctx)
        except Exception as exc:
            ctx.restore_config()
            log_path = write_log(
                test_name.replace("/", "_"),
                "basic test failed",
                [
                    ("exception", f"{type(exc).__name__}: {exc}"),
                    ("commands", ctx.command_log()),
                ],
            )
            results.append(BasicTestResult(test_name, False, str(exc), log_path))
            return results
        else:
            results.append(BasicTestResult(test_name, True, "passed"))
    return results
