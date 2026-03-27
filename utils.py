import os
import sys
import platform
import subprocess
import sysconfig
import io
import shlex
from typing import Optional

# get cpu architecture
def get_cpu_arch() -> str:
    arch = platform.machine()
    if "x86" in arch or "amd64" in arch:
        return "x86"
    elif "arm" in arch or "aarch64" in arch:
        return "arm"
    else:
        raise ValueError(f"❌ Unsupported architecture: {arch}")

# get device type
def get_device_type() -> str:
    import torch

    if torch.cuda.is_available():
        try:
            import ixformer
            return "ilu"
        except ImportError:
            return "cuda"

    try:
        import torch_musa
        if torch.musa.is_available():
            return "musa"
    except ImportError:
        pass

    try:
        import torch_mlu
        if torch.mlu.is_available():
            return "mlu"
    except ImportError:
        pass

    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass

    print("❌ Unsupported device type, please check what device you are using.")
    exit(1)

def get_base_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__))

def _join_path(*paths: str) -> str:
    return os.path.join(get_base_dir(), *paths)

# return the python version as a string like "310" or "311" etc
def get_python_version() -> str:
    return sysconfig.get_python_version().replace(".", "")

def get_torch_version(device: str) -> Optional[str]:
    try:
        import torch
        if device == "cuda":
            return torch.__version__
        return torch.__version__.split('+')[0]
    except ImportError:
        return None

def get_version() -> str:
    # first read from environment variable
    version: Optional[str] = os.getenv("XLLM_VERSION")
    if not version:
        # then read from version file
        with open(_join_path("version.txt"), "r") as f:
            version = f.read().strip()

    # strip the leading 'v' if present
    if version and version.startswith("v"):
        version = version[1:]

    if not version:
        raise RuntimeError("❌ Unable to find version string.")
    
    version_suffix = os.getenv("XLLM_VERSION_SUFFIX")
    if version_suffix:
        version += version_suffix
    return version

def read_readme() -> str:
    p = _join_path("README.md")
    if os.path.isfile(p):
        return io.open(p, "r", encoding="utf-8").read()
    else:
        return ""

def get_cmake_dir() -> str:
    plat_name = sysconfig.get_platform()
    python_version = get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = os.path.join(get_base_dir(), "build", dir_name)
    os.makedirs(cmake_dir, exist_ok=True)
    return cmake_dir

def check_and_install_pre_commit() -> None:
    # check if .git is a directory
    if not os.path.isdir(".git"):
        return
    
    if not os.path.exists(".git/hooks/pre-commit"):
        ok, _, _ = _run_command(["pre-commit", "install"], check=True)
        if not ok:
            print("❌ Run 'pre-commit install' failed. Please install pre-commit: pip install pre-commit")
            exit(1)

def _run_command(
    args: list[str],
    cwd: Optional[str] = None,
    check: bool = True,
    input_text: Optional[str] = None,
    passthrough_output: bool = False,
) -> tuple[bool, str, str]:
    try:
        if passthrough_output:
            result = subprocess.run(
                args,
                cwd=cwd,
                check=False,
                input=input_text,
                text=True,
            )
        else:
            result = subprocess.run(
                args,
                cwd=cwd,
                check=False,
                input=input_text,
                capture_output=True,
                text=True,
            )
    except OSError as e:
        return False, "", str(e)

    if passthrough_output:
        if check and result.returncode != 0:
            return False, "", f"exit code {result.returncode}"
        return result.returncode == 0, "", ""

    if check and result.returncode != 0:
        return False, result.stdout.strip(), (result.stderr or result.stdout).strip()

    return result.returncode == 0, result.stdout.strip(), (result.stderr or "").strip()

def _print_manual_check_commands(commands: list[str]) -> None:
    print("🔎 You can run these commands to inspect manually:")
    for cmd in commands:
        print(f"   {cmd}")

def _run_shell_command(
    command: str,
    cwd: Optional[str] = None,
    check: bool = True,
    passthrough_output: bool = False,
) -> bool:
    ok, _, err = _run_command(
        shlex.split(command),
        cwd=cwd,
        check=check,
        passthrough_output=passthrough_output,
    )
    if not ok:
        print(f"❌ Run shell command '{command}' failed: {err}")
        return False
    return True

def _run_git_command(repo_root: str, args: list[str]) -> tuple[bool, str]:
    ok, output, err = _run_command(["git"] + args, cwd=repo_root, check=True)
    if not ok and "No such file or directory" in err:
        print(f"❌ Failed to run git command in {repo_root}: git {' '.join(args)}")
        print(f"   {err}")
        return False, ""
    if not ok:
        print(f"❌ Git command failed in {repo_root}: git {' '.join(args)}")
        if err:
            print(f"   {err}")
        return False, ""
    return True, output

def _collect_submodule_init_issues(repo_root: str) -> dict[str, str]:
    ok, output = _run_git_command(repo_root, ["submodule", "status"])
    if not ok:
        print("❌ Failed to inspect submodule status.")
        _print_manual_check_commands([
            f"cd {repo_root}",
            "git submodule status",
            "git submodule update --init",
        ])
        exit(1)

    issues: dict[str, str] = {}
    for line in output.splitlines():
        if not line:
            continue
        state = line[0]
        content = line[1:].strip()
        parts = content.split()
        if len(parts) < 2:
            continue
        path = parts[1]
        commit = parts[0]

        if state == "-":
            issues[path] = f"uninitialized (expected commit starts with {commit})"
        elif state == "+":
            issues[path] = f"commit mismatch (checked-out commit starts with {commit})"
        elif state == "U":
            issues[path] = "merge conflict"

    return issues


def _install_dependencies(
    script_path: str,
    dependency: str,
    device: str,
    enable_ha: bool = False,
) -> None:
    enable_ha_flag = "true" if enable_ha else "false"
    command = f"sh third_party/dependencies.sh {dependency} {device} {enable_ha_flag}"
    if _run_shell_command(
        command,
        cwd=script_path,
        passthrough_output=True,
    ):
        return

    print(f"❌ Run shell command '{command}' failed!")
    _print_manual_check_commands([
        f"cd {script_path}",
        command,
    ])
    exit(1)


def _upsert_cmake_bool_option(option_name: str, enabled: bool) -> None:
    current_cmake_args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
    option_prefix = f"-D{option_name}="
    option_value = "ON" if enabled else "OFF"

    filtered_args = [arg for arg in current_cmake_args if not arg.startswith(option_prefix)]
    filtered_args.append(f"{option_prefix}{option_value}")
    os.environ["CMAKE_ARGS"] = " ".join(filtered_args)


def _validate_submodules_or_exit(repo_root: str) -> None:
    issues = _collect_submodule_init_issues(repo_root)

    if issues:
        print("❌ Submodule commit check failed. Repositories not correctly initialized:")
        for path in sorted(issues):
            print(f"   - {path}: {issues[path]}")
        print("\nPlease align submodules and try again:")
        print("   git submodule update --init [-f|--force]")
        exit(1)

def _get_cmake_cache_path() -> str:
    plat_name = sysconfig.get_platform()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{get_python_version()}"
    return os.path.join(get_base_dir(), "build", dir_name, "CMakeCache.txt")


def _get_xllm_ops_marker_path() -> str:
    ascend_home = os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    opp_root = os.path.join(ascend_home, "opp")
    return os.path.join(opp_root, "vendors", "xllm", ".xllm_ops_git_head")


def _clear_xllm_ops_cache_git_head(cache_path: str) -> bool:
    if not os.path.isfile(cache_path):
        return False

    cache_prefix = "XLLM_OPS_GIT_HEAD_CACHED:"
    with open(cache_path, "r", encoding="utf-8") as cache_file:
        old_lines = cache_file.readlines()

    new_lines = [line for line in old_lines if not line.startswith(cache_prefix)]
    if new_lines == old_lines:
        return False

    temp_file_path = f"{cache_path}.tmp"
    with open(temp_file_path, "w", encoding="utf-8") as cache_file:
        cache_file.writelines(new_lines)
    os.replace(temp_file_path, cache_path)
    return True


def _ensure_xllm_ops_rebuild_on_missing_marker() -> None:
    marker_path = _get_xllm_ops_marker_path()
    if os.path.isfile(marker_path):
        return

    cmake_cache_path = _get_cmake_cache_path()
    if _clear_xllm_ops_cache_git_head(cmake_cache_path):
        print("✅ Cleared XLLM_OPS_GIT_HEAD_CACHED from CMake cache to trigger xllm_ops rebuild.")
        return

def pre_build(device: Optional[str] = None, enable_ha: bool = False) -> None:
    script_path = os.path.dirname(os.path.abspath(__file__))
    normalized_device = (device or "").lower()

    _upsert_cmake_bool_option("ENABLE_HA", enable_ha)
    _validate_submodules_or_exit(script_path)
    _ensure_xllm_ops_rebuild_on_missing_marker()
    _install_dependencies(script_path, "yalantinglibs", normalized_device, enable_ha)
    if enable_ha:
        _install_dependencies(script_path, "mooncake", normalized_device, enable_ha)
