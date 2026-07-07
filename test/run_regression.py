#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "test"
ERROR_DIR = TEST_ROOT / "error"
DIFF_TOOL = TEST_ROOT / "tools" / "diff.py"
BUILD_SH = PROJECT_ROOT / "build.sh"
BUILD_BAT = PROJECT_ROOT / "build.bat"

MAX_MODEL_INDEX = 10

REQUIRED_CASE_FILES = [
    "model.pth",
    "model.py",
    "npu_config.py",
    "intr_move.json",
    "image.coe",
    "all.coe",
    "all.coe.map.txt",
    "instr.txt",
    "instr.asm",
]

GOLDEN_OUTPUTS = [
    ("all.coe", PROJECT_ROOT / "target" / "all.coe"),
    ("all.coe.map.txt", PROJECT_ROOT / "target" / "all.coe.map.txt"),
    ("instr.txt", PROJECT_ROOT / "data" / "instr.txt"),
    ("instr.asm", PROJECT_ROOT / "data" / "instr.asm"),
]


def parse_shell_build_var(text: str, name: str) -> str:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*[\"']?([^\"'\n#]+)[\"']?\s*(?:#.*)?$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"build.sh does not define {name}")
    return match.group(1).strip()


def parse_batch_build_var(text: str, name: str) -> str:
    pattern = rf"^\s*set\s+[\"']?{re.escape(name)}=([^\"'\n]+)[\"']?\s*$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError(f"build.bat does not define {name}")
    return match.group(1).strip()


def read_build_config(target_os: str) -> tuple[str, str]:
    if target_os == "windows":
        text = BUILD_BAT.read_text(encoding="utf-8")
        return parse_batch_build_var(text, "MODEL_NAME"), parse_batch_build_var(text, "MODEL_PY_NAME")
    text = BUILD_SH.read_text(encoding="utf-8")
    return parse_shell_build_var(text, "MODEL_NAME"), parse_shell_build_var(text, "MODEL_PY_NAME")


def build_command(target_os: str, clean: bool = False) -> list[str]:
    if target_os == "windows":
        command = ["cmd", "/c", "build.bat"]
    else:
        command = ["bash", "./build.sh"]
    if clean:
        command.append("clean")
    return command


def format_command(args: list[str]) -> str:
    return " ".join(args)


def discover_cases(max_index: int) -> list[Path]:
    return [TEST_ROOT / f"model{i}" for i in range(1, max_index + 1) if (TEST_ROOT / f"model{i}").is_dir()]


def missing_case_files(case_dir: Path) -> list[str]:
    return [name for name in REQUIRED_CASE_FILES if not (case_dir / name).is_file()]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_log(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def make_error_log_path(case_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ERROR_DIR / f"{case_dir.name}_{stamp}.log"


class WorkspaceBackup:
    def __init__(self, paths: Iterable[Path]) -> None:
        self.backup_dir = TEST_ROOT / ".tmp_regression_backup"
        self.paths = list(paths)
        self.records: list[tuple[Path, Path, bool]] = []

    def __enter__(self) -> "WorkspaceBackup":
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir(parents=True)
        for index, path in enumerate(self.paths):
            backup_path = self.backup_dir / f"{index}_{path.name}"
            existed = path.exists()
            if existed:
                copy_file(path, backup_path)
            self.records.append((path, backup_path, existed))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for path, backup_path, existed in self.records:
            if existed:
                copy_file(backup_path, path)
            elif path.exists():
                path.unlink()
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)


def install_case_inputs(case_dir: Path, model_name: str, model_py_name: str) -> None:
    copy_file(case_dir / "model.pth", PROJECT_ROOT / "model" / model_name)
    copy_file(case_dir / "model.py", PROJECT_ROOT / "model" / model_py_name)
    copy_file(case_dir / "npu_config.py", PROJECT_ROOT / "python" / "npu_config.py")
    copy_file(case_dir / "intr_move.json", PROJECT_ROOT / "data" / "intr_move.json")
    copy_file(case_dir / "image.coe", PROJECT_ROOT / "coe" / "image.coe")


def compare_outputs(case_dir: Path, log_lines: list[str]) -> bool:
    all_ok = True
    for golden_name, actual_path in GOLDEN_OUTPUTS:
        golden_path = case_dir / golden_name
        if not actual_path.is_file():
            log_lines.append(f"[MISSING OUTPUT] {actual_path}")
            all_ok = False
            continue

        result = run_command([sys.executable, str(DIFF_TOOL), str(golden_path), str(actual_path), "-n", "8"])
        if result.returncode == 0:
            log_lines.append(f"[OK] {golden_name}")
            continue

        all_ok = False
        log_lines.append(f"[DIFF] {golden_name} differs, returncode={result.returncode}")
        if result.stdout:
            log_lines.append("--- diff stdout ---")
            log_lines.append(result.stdout[:12000])
        if result.stderr:
            log_lines.append("--- diff stderr ---")
            log_lines.append(result.stderr[:4000])
    return all_ok


def run_case(case_dir: Path, model_name: str, model_py_name: str, target_os: str) -> bool:
    log_lines: list[str] = [
        f"case = {case_dir.name}",
        f"time = {datetime.now().isoformat(timespec='seconds')}",
        f"MODEL_NAME = {model_name}",
        f"MODEL_PY_NAME = {model_py_name}",
        f"target_os = {target_os}",
        "",
    ]

    missing = missing_case_files(case_dir)
    if missing:
        log_lines.append("[CASE FILE CHECK FAILED]")
        log_lines.extend(f"missing: {name}" for name in missing)
        log_path = make_error_log_path(case_dir)
        write_log(log_path, log_lines)
        print(f"{case_dir.name} test failed: missing files, see {log_path}")
        return False

    install_case_inputs(case_dir, model_name, model_py_name)

    clean_command = build_command(target_os, clean=True)
    clean = run_command(clean_command)
    log_lines.append(f"[COMMAND] {format_command(clean_command)}")
    log_lines.append(f"returncode = {clean.returncode}")
    if clean.stdout:
        log_lines.append("--- stdout ---")
        log_lines.append(clean.stdout)
    if clean.stderr:
        log_lines.append("--- stderr ---")
        log_lines.append(clean.stderr)
    if clean.returncode != 0:
        log_path = make_error_log_path(case_dir)
        write_log(log_path, log_lines)
        print(f"{case_dir.name} test failed: clean failed, see {log_path}")
        return False

    run_build_command = build_command(target_os, clean=False)
    build = run_command(run_build_command)
    log_lines.append(f"[COMMAND] {format_command(run_build_command)}")
    log_lines.append(f"returncode = {build.returncode}")
    if build.stdout:
        log_lines.append("--- stdout ---")
        log_lines.append(build.stdout)
    if build.stderr:
        log_lines.append("--- stderr ---")
        log_lines.append(build.stderr)
    if build.returncode != 0:
        log_path = make_error_log_path(case_dir)
        write_log(log_path, log_lines)
        print(f"{case_dir.name} test failed: build failed, see {log_path}")
        return False

    ok = compare_outputs(case_dir, log_lines)
    if ok:
        print(f"{case_dir.name} test passed")
        return True

    log_path = make_error_log_path(case_dir)
    write_log(log_path, log_lines)
    print(f"{case_dir.name} test failed: output differs, see {log_path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compiler golden regression tests in test/model1..model10.")
    parser.add_argument("--max-index", type=int, default=MAX_MODEL_INDEX, help="maximum modelN index to scan")
    parser.add_argument(
        "--os",
        choices=["windows", "linux"],
        default="windows",
        help="build script to use: windows -> build.bat, linux -> build.sh. Default: windows",
    )
    args = parser.parse_args()

    model_name, model_py_name = read_build_config(args.os)
    workspace_inputs = [
        PROJECT_ROOT / "model" / model_name,
        PROJECT_ROOT / "model" / model_py_name,
        PROJECT_ROOT / "python" / "npu_config.py",
        PROJECT_ROOT / "data" / "intr_move.json",
        PROJECT_ROOT / "coe" / "image.coe",
    ]

    cases = discover_cases(args.max_index)
    if not cases:
        print(f"No test cases found in {TEST_ROOT}/model1..model{args.max_index}")
        return 1

    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    with WorkspaceBackup(workspace_inputs):
        for case_dir in cases:
            if not run_case(case_dir, model_name, model_py_name, args.os):
                failed += 1

    total = len(cases)
    passed = total - failed
    print(f"Summary: {passed}/{total} test case(s) passed")
    if failed == 0:
        clean_command = build_command(args.os, clean=True)
        clean = run_command(clean_command)
        if clean.returncode != 0:
            log_path = ERROR_DIR / f"post_success_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            write_log(
                log_path,
                [
                    "post-success clean failed",
                    f"[COMMAND] {format_command(clean_command)}",
                    f"returncode = {clean.returncode}",
                    "--- stdout ---",
                    clean.stdout,
                    "--- stderr ---",
                    clean.stderr,
                ],
            )
            print(f"Post-success clean failed, see {log_path}")
            return 1
        print("Post-success clean completed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
