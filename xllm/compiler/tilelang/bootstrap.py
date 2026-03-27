from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from env import set_npu_envs

from .common.toolchain import (
    default_tilelang_root,
    git_head,
    prepare_tilelang_import,
    repo_root,
    resolve_tilelang_root,
)

PREPARE_ASCEND_COMMAND = "python xllm/compiler/tilelang_launcher.py prepare-ascend"


def _ready_error(message: str) -> RuntimeError:
    return RuntimeError(f"{message}\nRun `{PREPARE_ASCEND_COMMAND}` first.")


def _find_cann_set_env() -> Path | None:
    candidates: list[Path] = []
    npu_home_path = os.environ.get("NPU_HOME_PATH", "").strip()
    if npu_home_path:
        toolkit_root = Path(npu_home_path).resolve()
        candidates.append(toolkit_root / "set_env.sh")
        candidates.append(toolkit_root.parent / "set_env.sh")

    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
            Path("/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"),
        ]
    )

    for script in candidates:
        if script.is_file():
            return script.resolve()
    return None


def resolve_cann_set_env() -> Path:
    cann_set_env = _find_cann_set_env()
    if cann_set_env is not None:
        return cann_set_env

    set_npu_envs()
    cann_set_env = _find_cann_set_env()
    if cann_set_env is not None:
        return cann_set_env

    raise RuntimeError(
        "[ERROR] Cannot find CANN set_env.sh. Expected a path like "
        "/usr/local/Ascend/ascend-toolkit/set_env.sh."
    )


def ensure_tilelang_submodules(tilelang_root: str | Path) -> Path:
    tl_root = Path(tilelang_root).resolve()
    required_markers = {
        "3rdparty/catlass/CMakeLists.txt": tl_root / "3rdparty" / "catlass" / "CMakeLists.txt",
        "3rdparty/composable_kernel/CMakeLists.txt": (
            tl_root / "3rdparty" / "composable_kernel" / "CMakeLists.txt"
        ),
        "3rdparty/cutlass/CMakeLists.txt": tl_root / "3rdparty" / "cutlass" / "CMakeLists.txt",
        "3rdparty/pto-isa/CMakeLists.txt": tl_root / "3rdparty" / "pto-isa" / "CMakeLists.txt",
        "3rdparty/shmem/CMakeLists.txt": tl_root / "3rdparty" / "shmem" / "CMakeLists.txt",
        "3rdparty/tvm/CMakeLists.txt": tl_root / "3rdparty" / "tvm" / "CMakeLists.txt",
    }
    missing = [name for name, path in required_markers.items() if not path.is_file()]
    if missing:
        if (tl_root / ".git").exists():
            repair_hint = (
                "Run "
                f"`git -C {shlex.quote(str(tl_root))} submodule update --init --recursive` "
                "first."
            )
        else:
            bundled_root = default_tilelang_root().resolve()
            bundled_repair_cmd = (
                f"git -C {shlex.quote(str(repo_root()))} "
                "submodule update --init --recursive third_party/tilelang-ascend"
            )
            if tl_root == bundled_root:
                repair_hint = (
                    "Sync the bundled TileLang checkout from the xLLM repo root: "
                    f"`{bundled_repair_cmd}`."
                )
            else:
                repair_hint = (
                    f"`TL_ROOT={tl_root}` is not a git checkout. "
                    "Point TL_ROOT at a fully initialized tilelang-ascend clone, "
                    f"or sync the bundled checkout with `{bundled_repair_cmd}`."
                )
        raise RuntimeError(
            "[ERROR] tilelang-ascend nested dependencies are incomplete: "
            f"missing {', '.join(missing)}. "
            f"{repair_hint}"
        )
    return tl_root


def tilelang_git_head_cache_path(tilelang_root: str | Path) -> Path:
    return Path(tilelang_root).resolve() / "build" / ".xllm_tilelang_git_head_cached"


def read_tilelang_git_head_cached(tilelang_root: str | Path) -> str | None:
    cache_path = tilelang_git_head_cache_path(tilelang_root)
    if not cache_path.is_file():
        return None
    value = cache_path.read_text(encoding="utf-8").strip()
    return value or None


def write_tilelang_git_head_cached(tilelang_root: str | Path, head: str) -> None:
    cache_path = tilelang_git_head_cache_path(tilelang_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(head + "\n", encoding="utf-8")


def tilelang_artifacts_ready(tilelang_root: str | Path) -> bool:
    tl_root = Path(tilelang_root).resolve()
    required = [
        tl_root / "build" / "libtilelang_module.so",
        tl_root / "build" / "libtilelang.so",
        tl_root / "build" / "tvm" / "libtvm.so",
    ]
    return all(path.exists() for path in required)


def verify_tilelang_import(tilelang_root: str | Path) -> tuple[bool, str]:
    tl_root = prepare_tilelang_import(tilelang_root)
    env = os.environ.copy()
    env["TL_ROOT"] = str(tl_root)
    pythonpath = env.get("PYTHONPATH", "")
    pythonpath_items = [item for item in pythonpath.split(os.pathsep) if item]
    if str(tl_root) not in pythonpath_items:
        pythonpath_items.insert(0, str(tl_root))
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)
    cmd = [
        "bash",
        "-lc",
        "python - <<'PY'\n"
        "import tilelang\n"
        "print(getattr(tilelang, '__file__', '<unknown>'))\n"
        "PY",
    ]
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, detail
    return True, result.stdout.strip()


def _patch_tilelang_install_script(tilelang_root: str | Path) -> None:
    script_path = Path(tilelang_root).resolve() / "install_ascend.sh"
    if not script_path.is_file():
        raise RuntimeError("[ERROR] Missing tilelang install script: install_ascend.sh")

    lines = script_path.read_text(encoding="utf-8").splitlines()
    line_no = 145
    if len(lines) < line_no:
        raise RuntimeError(
            f"[ERROR] Unexpected install_ascend.sh: missing line {line_no}."
        )

    current_line = lines[line_no - 1].strip()
    if current_line == "make -j${MAKE_JOBS}":
        lines[line_no - 1] = "make -j"
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] Applied tilelang install parallel patch at line {line_no}: make -j")
        return

    if current_line == "make -j":
        return

    raise RuntimeError(
        f"[ERROR] Unexpected install_ascend.sh content at line {line_no}: {current_line!r}"
    )


def _run_tilelang_install(tilelang_root: str | Path, cann_set_env: str | Path) -> None:
    tl_root = ensure_tilelang_submodules(tilelang_root)
    _patch_tilelang_install_script(tl_root)

    cmd = (
        f"source {shlex.quote(str(cann_set_env))} && "
        "bash install_ascend.sh && "
        "source set_env.sh"
    )
    env = os.environ.copy()
    git_config_count = int(env.get("GIT_CONFIG_COUNT", "0") or "0")
    env[f"GIT_CONFIG_KEY_{git_config_count}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{git_config_count}"] = str(tl_root)
    env["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
    subprocess.check_call(
        ["bash", "-lc", cmd],
        cwd=str(tl_root),
        env=env,
    )


def ensure_ascend_ready() -> Path:
    set_npu_envs()
    tilelang_root = ensure_tilelang_submodules(resolve_tilelang_root())
    prepare_tilelang_import(tilelang_root)
    resolve_cann_set_env()

    if not tilelang_artifacts_ready(tilelang_root):
        raise _ready_error(
            f"[ERROR] tilelang-ascend artifacts are missing under {tilelang_root / 'build'}."
        )

    ok, detail = verify_tilelang_import(tilelang_root)
    if not ok:
        raise _ready_error(
            "[ERROR] Failed to import tilelang after configuring TL_ROOT="
            f"{tilelang_root}: {detail}"
        )

    return tilelang_root


def prepare_ascend(*, force: bool = False) -> Path:
    set_npu_envs()
    tilelang_root = ensure_tilelang_submodules(resolve_tilelang_root())
    prepare_tilelang_import(tilelang_root)
    cann_set_env = resolve_cann_set_env()

    current_head = git_head(tilelang_root)
    cached_head = read_tilelang_git_head_cached(tilelang_root)
    artifacts_ready = tilelang_artifacts_ready(tilelang_root)
    import_ok, import_detail = verify_tilelang_import(tilelang_root)

    install_reasons: list[str] = []
    if force:
        install_reasons.append("forced")
    if cached_head is None:
        install_reasons.append("HEAD cache missing")
    elif current_head != cached_head:
        install_reasons.append("HEAD changed")
    if not artifacts_ready:
        install_reasons.append("artifacts missing")
    if not import_ok:
        install_reasons.append("tilelang import failed")

    if install_reasons:
        print(
            "[INFO] Preparing tilelang-ascend: "
            + "; ".join(dict.fromkeys(install_reasons))
        )
        _run_tilelang_install(tilelang_root, cann_set_env)
        prepare_tilelang_import(tilelang_root)
        write_tilelang_git_head_cached(tilelang_root, current_head)

    if not tilelang_artifacts_ready(tilelang_root):
        raise RuntimeError(
            "[ERROR] tilelang-ascend artifacts are still missing after prepare."
        )

    import_ok, import_detail = verify_tilelang_import(tilelang_root)
    if not import_ok:
        raise RuntimeError(
            "[ERROR] tilelang import still failed after prepare: "
            f"{import_detail}"
        )

    print(f"[INFO] tilelang import success: {import_detail}")
    return tilelang_root
