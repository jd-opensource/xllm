#!/usr/bin/env python3

import io
import os
import re
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List
from jinja2 import Template

from distutils.core import Command
from setuptools import Extension, setup, find_packages
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

BUILD_TEST_FILE = True
BUILD_EXPORT = True

# get cpu architecture
def get_cpu_arch():
    arch = platform.machine()
    if "x86" in arch or "amd64" in arch:
        return "x86"
    elif "arm" in arch or "aarch64" in arch:
        return "arm"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

def get_cxx_abi():
    try:
        import torch
        return torch.compiled_with_cxx11_abi()
    except ImportError:
        return False


def get_base_dir():
    return os.path.abspath(os.path.dirname(__file__))


def join_path(*paths):
    return os.path.join(get_base_dir(), *paths)


def get_version():
    # first read from environment variable
    version = os.getenv("XLLM_VERSION")
    if not version:
        # then read from version file
        with open("version.txt", "r") as f:
            version = f.read().strip()

    # strip the leading 'v' if present
    if version and version.startswith("v"):
        version = version[1:]

    if not version:
        raise RuntimeError("Unable to find version string.")
    
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


def read_requirements() -> List[str]:
    file = join_path("cibuild/requirements.txt")
    with open(file) as f:
        return f.read().splitlines()


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version().replace(".", "")
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def get_python_include_path():
    try:
        from sysconfig import get_paths
        return get_paths()["include"]
    except ImportError:
        return None


def get_torch_root_path():
    try:
        import torch
        import os
        return os.path.dirname(os.path.abspath(torch.__file__))
    except ImportError:
        return None

def get_torch_mlu_root_path():
    try:
        import torch_mlu
        import os
        return os.path.dirname(os.path.abspath(torch_mlu.__file__))
    except ImportError:
        return None

def get_nccl_root_path():
    try:
        from nvidia import nccl
        return str(Path(nccl.__file__).parent)
    except ImportError:
        return None

def set_npu_envs():
    PYTORCH_NPU_INSTALL_PATH = os.getenv("PYTORCH_NPU_INSTALL_PATH")
    if not PYTORCH_NPU_INSTALL_PATH:
        os.environ["PYTORCH_NPU_INSTALL_PATH"] = "/usr/local/libtorch_npu"

    XLLM_KERNELS_PATH = os.getenv("XLLM_KERNELS_PATH")
    if not XLLM_KERNELS_PATH:
        os.environ["XLLM_KERNELS_PATH"] = "/usr/local/xllm_kernels"

    os.environ["PYTHON_INCLUDE_PATH"] = get_python_include_path()
    os.environ["PYTHON_LIB_PATH"] =  get_torch_root_path()
    os.environ["LIBTORCH_ROOT"] = get_torch_root_path()
    os.environ["INSTALL_XLLM_KERNELS"] = "ON" if install_kernels else "OFF"
    NPU_TOOLKIT_HOME = os.getenv("NPU_TOOLKIT_HOME")
    if not NPU_TOOLKIT_HOME:
        os.environ["NPU_TOOLKIT_HOME"] = "/usr/local/Ascend/ascend-toolkit/latest"
        NPU_TOOLKIT_HOME = "/usr/local/Ascend/ascend-toolkit/latest"
    LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH", "")
    arch = platform.machine()
    LD_LIBRARY_PATH = NPU_TOOLKIT_HOME+"/lib64" + ":" + \
        NPU_TOOLKIT_HOME+"/lib64/plugin/opskernel" + ":" + \
        NPU_TOOLKIT_HOME+"/lib64/plugin/nnengine" + ":" + \
        NPU_TOOLKIT_HOME+"/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/"+arch + ":" + \
        NPU_TOOLKIT_HOME+"/tools/aml/lib64" + ":" + \
        NPU_TOOLKIT_HOME+"/tools/aml/lib64/plugin" + ":" + \
        LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    PYTHONPATH = os.getenv("PYTHONPATH", "")
    PYTHONPATH = NPU_TOOLKIT_HOME+"/python/site-packages" + ":" + \
        NPU_TOOLKIT_HOME+"/opp/built-in/op_impl/ai_core/tbe" + ":" + \
        PYTHONPATH
    os.environ["PYTHONPATH"] = PYTHONPATH
    PATH = os.getenv("PATH", "")
    PATH = NPU_TOOLKIT_HOME+"/bin" + ":" + \
        NPU_TOOLKIT_HOME+"/compiler/ccec_compiler/bin" + ":" + \
        NPU_TOOLKIT_HOME+"/tools/ccec_compiler/bin" + ":" + \
        PATH
    os.environ["PATH"] = PATH
    os.environ["ASCEND_AICPU_PATH"] = NPU_TOOLKIT_HOME
    os.environ["ASCEND_OPP_PATH"] = NPU_TOOLKIT_HOME+"/opp"
    os.environ["TOOLCHAIN_HOME"] = NPU_TOOLKIT_HOME+"/toolkit"
    os.environ["NPU_HOME_PATH"] = NPU_TOOLKIT_HOME

    ATB_PATH = os.getenv("ATB_PATH")
    if not ATB_PATH:
        os.environ["ATB_PATH"] = "/usr/local/Ascend/nnal/atb"
        ATB_PATH = "/usr/local/Ascend/nnal/atb"


    ATB_HOME_PATH = ATB_PATH+"/latest/atb/cxx_abi_"+str(get_cxx_abi())
    LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH", "")
    LD_LIBRARY_PATH = ATB_HOME_PATH+"/lib" + ":" + \
        ATB_HOME_PATH+"/examples" + ":" + \
        ATB_HOME_PATH+"/tests/atbopstest" + ":" + \
        LD_LIBRARY_PATH
    os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    PATH = os.getenv("PATH", "")
    PATH = ATB_HOME_PATH+"/bin" + ":" + PATH
    os.environ["PATH"] = PATH

    os.environ["ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE"] = "0"
    os.environ["ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE"] = "0"
    os.environ["ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE"] = "0"
    os.environ["ATB_OPSRUNNER_SETUP_CACHE_ENABLE"] = "1"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_TYPE"] = "3"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT"] = "1"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT"] = "5"
    os.environ["ATB_OPSRUNNER_KERNEL_CACHE_TILING_SIZE"] = "10240"
    os.environ["ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE"] = "1"
    os.environ["ATB_WORKSPACE_MEM_ALLOC_GLOBAL"] = "0"
    os.environ["ATB_COMPARE_TILING_EVERY_KERNEL"] = "0"
    os.environ["ATB_HOST_TILING_BUFFER_BLOCK_NUM"] = "128"
    os.environ["ATB_DEVICE_TILING_BUFFER_BLOCK_NUM"] = "32"
    os.environ["ATB_SHARE_MEMORY_NAME_SUFFIX"] = ""
    os.environ["ATB_LAUNCH_KERNEL_WITH_TILING"] = "1"
    os.environ["ATB_MATMUL_SHUFFLE_K_ENABLE"] = "1"
    os.environ["ATB_RUNNER_POOL_SIZE"] = "64"
    os.environ["ASDOPS_HOME_PATH"] = ATB_HOME_PATH
    os.environ["ASDOPS_MATMUL_PP_FLAG"] = "1"
    os.environ["ASDOPS_LOG_LEVEL"] = "ERROR"
    os.environ["ASDOPS_LOG_TO_STDOUT"] = "0"
    os.environ["ASDOPS_LOG_TO_FILE"] = "1"
    os.environ["ASDOPS_LOG_TO_FILE_FLUSH"] = "0"
    os.environ["ASDOPS_LOG_TO_BOOST_TYPE"] = "atb"
    os.environ["ASDOPS_LOG_PATH"] = "~"
    os.environ["ASDOPS_TILING_PARSE_CACHE_DISABLE"] = "0"
    os.environ["LCCL_DETERMINISTIC"] = "0"
    os.environ["LCCL_PARALLEL"] = "0"


def set_mlu_envs():
    os.environ["PYTHON_INCLUDE_PATH"] = get_python_include_path()
    os.environ["PYTHON_LIB_PATH"] =  get_torch_root_path()
    os.environ["LIBTORCH_ROOT"] = get_torch_root_path()
    os.environ["PYTORCH_INSTALL_PATH"] = get_torch_root_path()
    os.environ["PYTORCH_MLU_INSTALL_PATH"] = get_torch_mlu_root_path()
    
def set_cuda_envs():
    os.environ["PYTHON_INCLUDE_PATH"] = get_python_include_path()
    os.environ["PYTHON_LIB_PATH"] =  get_torch_root_path()
    os.environ["LIBTORCH_ROOT"] = get_torch_root_path()
    os.environ["PYTORCH_INSTALL_PATH"] = get_torch_root_path()
    os.environ["CUDA_TOOLKIT_ROOT_DIR"] = "/usr/local/cuda"
    os.environ["NCCL_ROOT"] = get_nccl_root_path()
    os.environ["NCCL_VERSION"] = "2"
    
class CMakeExtension(Extension):
    def __init__(self, name: str, path: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.path = path


class ExtBuild(build_ext):
    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of xLLM project"),
        ("device=", None, "target device type (a3 or a2 or mlu or cuda)"),
        ("arch=", None, "target arch type (x86 or arm)"),
        ("install-xllm-kernels=", None, "install xllm_kernels RPM package (true/false)"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()
        self.device = None  
        self.arch = None
        self.install_xllm_kernels = None

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        # check if cmake is installed
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
            exit(1)

        match = re.search(
            r"version\s*(?P<major>\d+)\.(?P<minor>\d+)([\d.]+)?", out.decode()
        )
        cmake_major, cmake_minor = int(match.group("major")), int(match.group("minor"))
        if (cmake_major, cmake_minor) < (3, 18):
            raise RuntimeError("CMake >= 3.18.0 is required")

        try:
            # build extensions
            for ext in self.extensions:
                self.build_extension(ext)
        except Exception as e:
            print("Build failed.")
            exit(1)

    def build_extension(self, ext: CMakeExtension):
        ninja_dir = shutil.which("ninja")
        # the output dir for the extension
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))

        # create build directory
        os.makedirs(self.build_temp, exist_ok=True)

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        build_type = "Debug" if debug else "Release"

        cmake_args = [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            "-DUSE_CCACHE=ON",
            "-DUSE_MANYLINUX:BOOL=ON",
            f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DBUILD_SHARED_LIBS=OFF",
            f"-DDEVICE_TYPE=USE_{self.device.upper()}",
            f"-DDEVICE_ARCH={self.arch.upper()}",
            f"-DINSTALL_XLLM_KERNELS={'ON' if self.install_xllm_kernels else 'OFF'}",
        ]
        
        if self.device == "a2" or self.device == "a3":
            cmake_args += ["-DUSE_NPU=ON"]
            # set npu environment variables
            set_npu_envs()
        elif self.device == "mlu":
            cmake_args += ["-DUSE_MLU=ON"]
            # set mlu environment variables
            set_mlu_envs()
        elif self.device == "cuda":
            cuda_architectures = "80;89;90"
            cmake_args += ["-DUSE_CUDA=ON", 
                           f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}"]
            # set cuda environment variables
            set_cuda_envs()
        else:
            raise ValueError("Please set --device to a2 or a3 or mlu or cuda.")


        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # check if torch binary is built with cxx11 abi
        if get_cxx_abi():
            cmake_args += ["-DUSE_CXX11_ABI=ON"]
        else:
            cmake_args += ["-DUSE_CXX11_ABI=OFF"]
        
        build_args = ["--config", build_type]
        max_jobs = os.getenv("MAX_JOBS", str(os.cpu_count()))
        build_args += ["-j" + max_jobs]

        env = os.environ.copy()
        print("CMake Args: ", cmake_args)
        print("Env: ", env)

        cmake_dir = get_cmake_dir()
        subprocess.check_call(
            ["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env
        )

        base_build_args = build_args
        # add build target to speed up the build process
        build_args += ["--target", ext.name, "xllm"]
        subprocess.check_call(["cmake", "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

        os.makedirs(os.path.join(os.path.dirname(cmake_dir), "xllm/core/server/"), exist_ok=True)
        shutil.copy(
            os.path.join(extdir, "xllm"),
            os.path.join(os.path.dirname(cmake_dir), "xllm/core/server/"),
        )

        if BUILD_EXPORT:
            # build export module
            build_args = base_build_args + ["--target export_module"]
            subprocess.check_call(["cmake", "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

        if BUILD_TEST_FILE:
            # build tests target
            build_args = base_build_args + ["--target all_tests"]
            subprocess.check_call(["cmake", "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

class BuildDistWheel(bdist_wheel):
    user_options = bdist_wheel.user_options + [
        ("device=", None, "target device type (a3 or a2 or mlu or cuda)"),
        ("arch=", None, "target arch type (x86 or arm)"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.device = None
        self.arch = None

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        build_ext_cmd = self.get_finalized_command('build_ext')
        build_ext_cmd.device = self.device
        build_ext_cmd.arch = self.arch
        self.run_command('build_ext')
        if self.arch == 'arm':
            ext_path = get_base_dir() + "/build/lib.linux-aarch64-cpython-311/"
        else:
            ext_path = get_base_dir() + "/build/lib.linux-x86_64-cpython-311/"
        if len(ext_path) == 0:
            print("Build wheel failed, not found path.")
            exit(1)
        tmp_path = os.path.join(ext_path, 'xllm')
        for root, dirs, files in os.walk(tmp_path):
            for item in files:
                path = os.path.join(root, item)
                if '_test' in item and os.path.isfile(path):
                    os.remove(path)
        global BUILD_TEST_FILE
        BUILD_TEST_FILE = False

        self.skip_build = True
        super().run()

class TestUT(Command):
    description = "Run all testing binary."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run_ctest(self, cmake_dir):
        try:
            '''
            result = subprocess.run(
                ['ctest'],
                check=True,
                capture_output=True,
                text=True,
                cwd=cmake_dir
            )
            print(result.stdout)
            '''
            process = subprocess.Popen(
                ['ctest', '--parallel', '8'],
                cwd=cmake_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

            return_code = process.wait()
            if return_code != 0:
              print("Testing failed.")
              exit(1)
            return return_code
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            exit(1)

    def run(self):
        self.run_ctest(get_cmake_dir())

def check_and_install_pre_commit():
    # check if .git is a directory
    if not os.path.isdir(".git"):
        return
    
    if not os.path.exists(".git/hooks/pre-commit"):
        os.system("pre-commit install")
        if not os.path.exists(".git/hooks/pre-commit"):
            print("Run 'pre-commit install' failed. Please install pre-commit: pip install pre-commit")
            exit(0)

def run_git_command(command, cwd=None, check=True):
    try:
        subprocess.run(command, cwd=cwd, check=check, shell=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr.strip()}")
        return False

def has_uncommitted_changes(repo_path):
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())

def is_safe_directory_set(repo_path):
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

def apply_patch_safely(patch_file_path, repo_path):
    print(f"🔍 Checking repo status: {repo_path}")

    if not is_safe_directory_set(repo_path):
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", repo_path],
                check=True
            )
            print(f"✅ Add safe.directory success: {repo_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Add safe.directory fail: {e.stderr}")
            print(f"   Please manually set 'git config --global --add safe.directory {repo_path}'")
            return False

    if has_uncommitted_changes(repo_path):
        print(f"⚠️ Uncommitted changes detected. Running `git reset --hard` for {repo_path}")
        if not run_git_command("git reset --hard", cwd=repo_path):
            print("❌ Failed to reset changes!")
            return False
    
    print(f"🛠️ Apply patch: {patch_file_path}")
    apply_success = run_git_command(f"git apply --check {patch_file_path}", cwd=repo_path, check=False)
    
    if apply_success:
        if not run_git_command(f"git apply {patch_file_path}", cwd=repo_path):
            print("❌ apply patch fail!")
            apply_success = False
    
    if apply_success:
        print("🎉 Success apply patch!")
        return True
    else:
        print("\n❌ Conflicts detected! Please resolve manually and retry:")
        print(f"  cd {repo_path} && git apply {patch_file_path}")
        return False

def apply_patch():
    if os.path.exists("third_party/custom_patch"):
        script_path = os.path.dirname(os.path.abspath(__file__))
        mooncake_repo_path = os.path.join(script_path, "third_party/Mooncake")
        if not apply_patch_safely("../custom_patch/Mooncake.patch", mooncake_repo_path):
            exit(0)
        cpprestsdk_repo_path = os.path.join(script_path, "third_party/cpprestsdk")
        if not apply_patch_safely("../custom_patch/cpprestsdk.patch", cpprestsdk_repo_path):
            exit(0)

if __name__ == "__main__":
    device = 'a2'  # default
    arch = get_cpu_arch()
    install_kernels = True
    if '--device' in sys.argv:
        idx = sys.argv.index('--device')
        if idx + 1 < len(sys.argv):
            device = sys.argv[idx+1].lower()
            if device not in ('a2', 'a3', 'mlu', 'cuda'):
                print("Error: --device must be a2 or a3 or mlu (case-insensitive)")
                sys.exit(1)
            # Remove the arguments so setup() doesn't see them
            del sys.argv[idx]
            del sys.argv[idx]
    if '--dry_run' not in sys.argv:
        apply_patch()
    else:
        sys.argv.remove("--dry_run") 
    if '--install-xllm-kernels' in sys.argv:
        idx = sys.argv.index('--install-xllm-kernels')
        if idx + 1 < len(sys.argv):
            install_kernels = sys.argv[idx+1].lower()
            if install_kernels in ('true', '1', 'yes', 'y', 'on'):
                install_kernels = True
            elif install_kernels in ('false', '0', 'no', 'n', 'off'):
                install_kernels = False
            else:
                print("Error: --install-xllm-kernels must be true or false")
                sys.exit(1)
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    if "SKIP_TEST" in os.environ:
        BUILD_TEST_FILE = False
    if "SKIP_EXPORT" in os.environ:
        BUILD_EXPORT = False
    
    version = get_version()

    # check and install git pre-commit
    check_and_install_pre_commit()

    setup(
        name="xllm",
        version=version,
        license="Apache 2.0",
        author="xLLM Team",
        author_email="infer@jd.com",
        description="A high-performance inference system for large language models.",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/jd-opensource/xllm",
        project_url={
            "Homepage": "https://xllm.readthedocs.io/zh-cn/latest/",
            "Documentation": "https://xllm.readthedocs.io/zh-cn/latest/",
        },
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Environment :: NPU",
            "Operating System :: POSIX",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        ext_modules=[CMakeExtension("xllm", "xllm/")],
        cmdclass={"build_ext": ExtBuild,
                  "test": TestUT,
                  'bdist_wheel': BuildDistWheel},
        options={'build_ext': {
                    'device': device,
                    'arch': arch,
                    'install_xllm_kernels': install_kernels if install_kernels is not None else "false"
                    },
                 'bdist_wheel': {
                    'device': device,
                    'arch': arch,
                    }
                },
        zip_safe=False,
        py_modules=["xllm/launch_xllm", "xllm/__init__",
                    "xllm/pybind/llm", "xllm/pybind/vlm",
                    "xllm/pybind/util", "xllm/pybind/args"],
        entry_points={
            'console_scripts': [
                'xllm = xllm.launch_xllm:launch_xllm'
            ],
        },
        python_requires=">=3.8",
        #install_requires=read_requirements(),
    )
