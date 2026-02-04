import os
import sys
import platform
import subprocess
import sysconfig
import io
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
            return "a2"
    except ImportError:
        pass

    print("❌ Unsupported device type, please check what device you are using.")
    exit(1)

def get_base_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__))

def join_path(*paths: str) -> str:
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
        with open(join_path("version.txt"), "r") as f:
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
    p = join_path("README.md")
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
        try:
            subprocess.run(["pre-commit", "install"], check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Run 'pre-commit install' failed. Please install pre-commit: pip install pre-commit")
            exit(1)

def run_shell_command(command: str, cwd: Optional[str] = None, check: bool = True) -> bool:
    try:
        import shlex
        subprocess.run(shlex.split(command), cwd=cwd, check=check, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Run shell command '{command}' failed: {e.stderr.strip() if hasattr(e, 'stderr') else e}")
        return False

def has_uncommitted_changes(repo_path: str) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())

def is_safe_directory_set(repo_path: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "config", "--global", "--get-all", "safe.directory"],
            capture_output=True,
            text=True,
            check=True
        )
        existing_paths = result.stdout.strip().split("\n")
        return repo_path in existing_paths
    except subprocess.CalledProcessError:
        return False 

def apply_patch_safely(patch_file_path: str, repo_path: str) -> bool:
    print(f"🔍 Checking repo status: {repo_path}")

    if not is_safe_directory_set(repo_path):
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", repo_path],
                check=True
            )
            print(f"✅ Add safe.directory success: {repo_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Add safe.directory fail: {e.stderr.strip()}")
            print(f"   Please manually set 'git config --global --add safe.directory {repo_path}'")
            return False

    if has_uncommitted_changes(repo_path):
        print(f"⚠️ Uncommitted changes detected. Running `git reset --hard` for {repo_path}")
        if not run_shell_command("git reset --hard", cwd=repo_path):
            print("❌ Failed to reset changes!")
            return False
    
    print(f"🛠️ Apply patch: {patch_file_path}")
    apply_success = run_shell_command(f"git apply --check {patch_file_path}", cwd=repo_path, check=False)
    
    if apply_success:
        if not run_shell_command(f"git apply {patch_file_path}", cwd=repo_path):
            print(f"❌ Apply patch '{patch_file_path}' failed!")
            apply_success = False
    
    if apply_success:
        print("🎉 Success apply patch!")
        return True
    else:
        print("\n❌ Conflicts detected! Please resolve manually and retry:")
        print(f"  cd {repo_path} && git apply {patch_file_path}")
        return False

def pre_build(device: str) -> None:
    if os.path.exists("third_party/custom_patch"):
        script_path = os.path.dirname(os.path.abspath(__file__))
        mooncake_repo_path = os.path.join(script_path, "third_party/Mooncake")
        if device in ("a2", "a3"):
            if not apply_patch_safely("../custom_patch/Mooncake_npu.patch", mooncake_repo_path):
                print("❌ Failed to apply Mooncake_npu.patch!")
                exit(1)
        else:
            if not apply_patch_safely("../custom_patch/Mooncake.patch", mooncake_repo_path):
                print("❌ Failed to apply Mooncake.patch!")
                exit(1)

        cpprestsdk_repo_path = os.path.join(script_path, "third_party/cpprestsdk")
        if not apply_patch_safely("../custom_patch/cpprestsdk.patch", cpprestsdk_repo_path):
            print("❌ Failed to apply cpprestsdk.patch!")
            exit(1)

        if not run_shell_command("sh third_party/dependencies.sh", cwd=script_path):
            print("❌ Run shell command 'sh third_party/dependencies.sh' failed!")
            exit(1)

        # export CMAKE_PREFIX_PATH to environment for yalantinglibs or other dependencies
        os.environ["CMAKE_PREFIX_PATH"] = os.path.join(os.path.expanduser("~"), ".local")
        print(f"✅ Export CMAKE_PREFIX_PATH to environment: {os.environ['CMAKE_PREFIX_PATH']}")
