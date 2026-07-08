#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test"
ERROR_DIR = TEST_ROOT / "error"
PYTHON_CONFIG = PROJECT_ROOT / "python" / "npu_config.py"
TARGET_COE = PROJECT_ROOT / "target" / "all.coe"
DIFF_TOOL = TEST_ROOT / "tools" / "diff.py"
IGNORED_DIRS = {"tools", "error", "__pycache__"}
REQUIRED_FILES = ("model.pt2", "npu_config.py", "intr_move.json", "all.coe")


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
class TestResult:
    name: str
    passed: bool
    message: str
    log_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixture-based compiler regression tests.")
    parser.add_argument(
        "--os",
        choices=("windows", "linux"),
        default="windows",
        help="select build script: windows uses build.bat, linux uses build.sh",
    )
    return parser.parse_args()


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


def discover_fixtures() -> list[Path]:
    fixtures = []
    for path in sorted(TEST_ROOT.iterdir()):
        if path.is_dir() and path.name not in IGNORED_DIRS:
            fixtures.append(path)
    return fixtures


def missing_required_files(fixture: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (fixture / name).is_file()]


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


def format_command_result(result: CommandResult) -> str:
    return (
        f"command: {result.command_text}\n"
        f"returncode: {result.returncode}\n\n"
        f"[stdout]\n{result.stdout}\n\n"
        f"[stderr]\n{result.stderr}"
    )


def first_different_line(expected: Path, actual: Path) -> str:
    expected_lines = expected.read_text(encoding="utf-8", errors="replace").splitlines()
    actual_lines = actual.read_text(encoding="utf-8", errors="replace").splitlines()
    limit = max(len(expected_lines), len(actual_lines))
    for index in range(limit):
        expected_line = expected_lines[index] if index < len(expected_lines) else "<missing>"
        actual_line = actual_lines[index] if index < len(actual_lines) else "<missing>"
        if expected_line != actual_line:
            return (
                f"first_different_line: {index + 1}\n"
                f"expected: {expected_line}\n"
                f"actual  : {actual_line}"
            )
    return "files have identical text lines"


def copy_fixture_config(fixture: Path) -> None:
    shutil.copy2(fixture / "npu_config.py", PYTHON_CONFIG)


def restore_config(original_config: bytes) -> None:
    PYTHON_CONFIG.write_bytes(original_config)


def run_fixture(fixture: Path, os_name: str) -> TestResult:
    test_name = fixture.name
    missing = missing_required_files(fixture)
    if missing:
        log_path = write_log(
            test_name,
            "fixture file check failed",
            [
                ("missing files", "\n".join(missing)),
                ("fixture path", str(fixture)),
                ("required files", "\n".join(REQUIRED_FILES)),
            ],
        )
        return TestResult(test_name, False, f"missing required files: {', '.join(missing)}", log_path)

    clean_before = run_command(build_command(os_name, "clean"))
    if clean_before.returncode != 0:
        log_path = write_log(
            test_name,
            "pre-build clean failed",
            [("clean result", format_command_result(clean_before))],
        )
        return TestResult(test_name, False, "pre-build clean failed", log_path)

    copy_fixture_config(fixture)

    build_result = run_command(build_command(os_name))
    if build_result.returncode != 0:
        log_path = write_log(
            test_name,
            "build failed",
            [
                ("fixture", str(fixture)),
                ("build result", format_command_result(build_result)),
            ],
        )
        return TestResult(test_name, False, "build failed", log_path)

    golden = fixture / "all.coe"
    diff_result = run_command(["python", str(DIFF_TOOL), str(golden), str(TARGET_COE), "-n", "4"])
    if diff_result.returncode != 0:
        line_info = first_different_line(golden, TARGET_COE) if TARGET_COE.is_file() else "target/all.coe was not generated"
        map_hint = []
        fixture_map = fixture / "all.coe.map.txt"
        if fixture_map.is_file():
            map_hint.append(f"fixture map: {fixture_map}")
        generated_map = PROJECT_ROOT / "target" / "all.coe.map.txt"
        if generated_map.is_file():
            map_hint.append(f"generated map: {generated_map}")
        log_path = write_log(
            test_name,
            "golden COE mismatch",
            [
                ("line mismatch", line_info),
                ("diff result", format_command_result(diff_result)),
                ("map files", "\n".join(map_hint) if map_hint else "no map files available"),
            ],
        )
        return TestResult(test_name, False, "golden COE mismatch", log_path)

    return TestResult(test_name, True, "passed")


def main() -> int:
    args = parse_args()
    fixtures = discover_fixtures()
    if not fixtures:
        print("No regression fixtures found under ./test")
        return 1

    original_config = PYTHON_CONFIG.read_bytes()
    results: list[TestResult] = []

    try:
        for fixture in fixtures:
            print(f"[RUN] {fixture.name}")
            result = run_fixture(fixture, args.os)
            results.append(result)
            if result.passed:
                print(f"[PASS] {result.name}")
            else:
                print(f"[FAIL] {result.name}: {result.message}")
                if result.log_path is not None:
                    print(f"       log: {result.log_path}")
    finally:
        restore_config(original_config)
        final_clean = run_command(build_command(args.os, "clean"))
        if final_clean.returncode != 0:
            log_path = write_log(
                "final_cleanup",
                "final clean failed",
                [("clean result", format_command_result(final_clean))],
            )
            print(f"[WARN] final clean failed, log: {log_path}")

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    print(f"[SUMMARY] passed={passed} failed={failed} total={len(results)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
