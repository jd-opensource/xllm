from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.build_support.env import set_npu_envs

from .common.toolchain import (
    default_tilelang_root,
    git_head,
    prepare_tilelang_import,
    repo_root,
    resolve_tilelang_root,
)

PREPARE_ASCEND_COMMAND = "python xllm/compiler/tilelang_launcher.py prepare-ascend"


@dataclass(frozen=True)
class TilelangPrepareState:
    tilelang_root: Path
    cann_set_env: Path
    current_head: str
    cached_head: str | None
    artifacts_ready: bool
    import_ok: bool
    import_detail: str


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


def _tilelang_patch_dir() -> Path:
    return Path(__file__).resolve().parent / "patches" / "tilelang_ascend"


def _tilelang_install_patch_path() -> Path:
    patch_path = _tilelang_patch_dir() / "0001-install-ascend.patch"
    if not patch_path.is_file():
        raise RuntimeError(f"[ERROR] Missing TileLang patch file: {patch_path}")
    return patch_path


def _git_apply_base_cmd(repo_root: Path) -> list[str]:
    return [
        "git",
        "-c",
        f"safe.directory={repo_root}",
        "-C",
        str(repo_root),
        "apply",
        "--whitespace=nowarn",
    ]


def _check_git_patch_state(repo_root: Path, patch_path: Path) -> str:
    apply_check = subprocess.run(
        _git_apply_base_cmd(repo_root) + ["--check", str(patch_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if apply_check.returncode == 0:
        return "unapplied"

    reverse_check = subprocess.run(
        _git_apply_base_cmd(repo_root) + ["--reverse", "--check", str(patch_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if reverse_check.returncode == 0:
        return "applied"

    apply_detail = (apply_check.stderr or apply_check.stdout).strip()
    reverse_detail = (reverse_check.stderr or reverse_check.stdout).strip()
    raise RuntimeError(
        "[ERROR] Failed to match TileLang patch "
        f"{patch_path.name} against {repo_root}.\n"
        f"apply --check: {apply_detail or '<no output>'}\n"
        f"reverse --check: {reverse_detail or '<no output>'}"
    )


def _apply_git_patch(repo_root: Path, patch_path: Path, message: str) -> None:
    if _check_git_patch_state(repo_root, patch_path) == "applied":
        return
    subprocess.check_call(_git_apply_base_cmd(repo_root) + [str(patch_path)])
    print(message)


def _restore_git_patch(repo_root: Path, patch_path: Path) -> None:
    if _check_git_patch_state(repo_root, patch_path) != "applied":
        return
    subprocess.check_call(_git_apply_base_cmd(repo_root) + ["--reverse", str(patch_path)])


def _patch_tilelang_install_tree(tilelang_root: str | Path) -> None:
    tl_root = Path(tilelang_root).resolve()
    required_files = (
        tl_root / "install_ascend.sh",
        tl_root / "requirements-build.txt",
    )
    missing = [str(path.name) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            "[ERROR] Missing tilelang install files: " + ", ".join(missing)
        )
    _apply_git_patch(
        tl_root,
        _tilelang_install_patch_path(),
        "[INFO] Applied tilelang install patch",
    )


def _restore_tilelang_install_tree(tilelang_root: str | Path) -> None:
    tl_root = Path(tilelang_root).resolve()
    _restore_git_patch(tl_root, _tilelang_install_patch_path())


def _run_tilelang_install(tilelang_root: str | Path, cann_set_env: str | Path) -> None:
    tl_root = ensure_tilelang_submodules(tilelang_root)
    _patch_tilelang_install_tree(tl_root)

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
    try:
        subprocess.check_call(
            ["bash", "-lc", cmd],
            cwd=str(tl_root),
            env=env,
        )
    finally:
        _restore_tilelang_install_tree(tl_root)


def collect_prepare_state() -> TilelangPrepareState:
    tilelang_root = ensure_tilelang_submodules(resolve_tilelang_root())
    prepare_tilelang_import(tilelang_root)
    return TilelangPrepareState(
        tilelang_root=tilelang_root,
        cann_set_env=resolve_cann_set_env(),
        current_head=git_head(tilelang_root),
        cached_head=read_tilelang_git_head_cached(tilelang_root),
        artifacts_ready=tilelang_artifacts_ready(tilelang_root),
        import_ok=False,
        import_detail="",
    )


def refresh_prepare_state_import(state: TilelangPrepareState) -> TilelangPrepareState:
    import_ok, import_detail = verify_tilelang_import(state.tilelang_root)
    return TilelangPrepareState(
        tilelang_root=state.tilelang_root,
        cann_set_env=state.cann_set_env,
        current_head=state.current_head,
        cached_head=state.cached_head,
        artifacts_ready=tilelang_artifacts_ready(state.tilelang_root),
        import_ok=import_ok,
        import_detail=import_detail,
    )


def prepare_state() -> TilelangPrepareState:
    return refresh_prepare_state_import(collect_prepare_state())


def install_reasons(state: TilelangPrepareState, *, force: bool) -> list[str]:
    reasons: list[str] = []
    if force:
        reasons.append("forced")
    if state.cached_head is None:
        reasons.append("HEAD cache missing")
    elif state.current_head != state.cached_head:
        reasons.append("HEAD changed")
    if not state.artifacts_ready:
        reasons.append("artifacts missing")
    if not state.import_ok:
        reasons.append("tilelang import failed")
    return list(dict.fromkeys(reasons))


def ensure_ascend_ready() -> Path:
    set_npu_envs()
    state = prepare_state()

    if not state.artifacts_ready:
        raise _ready_error(
            "[ERROR] tilelang-ascend artifacts are missing under "
            f"{state.tilelang_root / 'build'}."
        )

    if not state.import_ok:
        raise _ready_error(
            "[ERROR] Failed to import tilelang after configuring TL_ROOT="
            f"{state.tilelang_root}: {state.import_detail}"
        )

    return state.tilelang_root


def prepare_ascend(*, force: bool = False) -> Path:
    set_npu_envs()
    state = prepare_state()
    reasons = install_reasons(state, force=force)

    if reasons:
        print("[INFO] Preparing tilelang-ascend: " + "; ".join(reasons))
        _run_tilelang_install(state.tilelang_root, state.cann_set_env)
        prepare_tilelang_import(state.tilelang_root)
        write_tilelang_git_head_cached(state.tilelang_root, state.current_head)
        state = prepare_state()

    if not state.artifacts_ready:
        raise RuntimeError(
            "[ERROR] tilelang-ascend artifacts are still missing after prepare."
        )

    if not state.import_ok:
        raise RuntimeError(
            "[ERROR] tilelang import still failed after prepare: "
            f"{state.import_detail}"
        )

    print(f"[INFO] tilelang import success: {state.import_detail}")
    return state.tilelang_root
