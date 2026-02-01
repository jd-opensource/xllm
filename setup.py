import io
import os
import re
import shutil
import subprocess
import sys
import argparse

from distutils.core import Command
from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext

from env import get_cxx_abi, set_npu_envs, set_mlu_envs, set_cuda_envs, set_ilu_envs, set_musa_envs
from utils import get_cpu_arch, get_device_type, pre_build, get_version, check_and_install_pre_commit, read_readme, get_cmake_dir, get_base_dir, get_python_version, get_torch_version

BUILD_TEST_FILE = True
BUILD_EXPORT = True
        
class CMakeExtension(Extension):
    def __init__(self, name: str, path: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.realpath(os.path.abspath(sourcedir))
        self.path = path

class ExtBuild(build_ext):
    user_options = build_ext.user_options + [
        ("base-dir=", None, "base directory of xLLM project"),
        ("device=", None, "target device type (a3 or a2 or mlu or cuda or musa)"),
        ("arch=", None, "target arch type (x86 or arm)"),
        ("install-xllm-kernels=", None, "install xllm_kernels RPM package (true/false)"),
        ("generate-so=", None, "generate so or binary"),
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = get_base_dir()
        self.device = None  
        self.arch = None
        self.install_xllm_kernels = None
        self.generate_so = False

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
            print("ERROR: Build failed.")
            print(f"Details: {e}")
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

        max_jobs = os.getenv("MAX_JOBS", str(os.cpu_count()))
        max_jobs_int = int(max_jobs)
        
        # Limit archive (ar/ranlib) concurrency to avoid file locking conflicts.
        # The ar tool requires exclusive access to archive files (.a files) when
        # creating or updating static libraries. When multiple ar processes attempt
        # to modify the same archive file simultaneously, they compete for file locks,
        # which can cause deadlocks and hang the build process.
        archive_jobs = min(8, max(1, max_jobs_int // 4))
        cmake_args = [
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            "-DUSE_CCACHE=ON",
            f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DBUILD_SHARED_LIBS=OFF",
            f"-DDEVICE_TYPE=USE_{self.device.upper()}",
            f"-DDEVICE_ARCH={self.arch.upper()}",
            f"-DINSTALL_XLLM_KERNELS={'ON' if self.install_xllm_kernels else 'OFF'}",
            f"-DCMAKE_JOB_POOLS=archive={archive_jobs}",
        ]

        if self.device == "a2" or self.device == "a3":
            cmake_args += ["-DUSE_NPU=ON"]
            set_npu_envs()
        elif self.device == "mlu":
            cmake_args += ["-DUSE_MLU=ON"]
            set_mlu_envs()
        elif self.device == "cuda":
            torch_cuda_architectures = os.getenv("TORCH_CUDA_ARCH_LIST")
            if not torch_cuda_architectures:
                raise ValueError("Please set TORCH_CUDA_ARCH_LIST environment variable, e.g. export TORCH_CUDA_ARCH_LIST=\"8.0 8.9 9.0 10.0 12.0\"")
            cmake_args += ["-DUSE_CUDA=ON", 
                           f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_architectures}"]
            set_cuda_envs()
        elif self.device == "ilu":
            cmake_args += ["-DUSE_ILU=ON"]
            set_ilu_envs()
        elif self.device == "musa":
            cmake_args += ["-DUSE_MUSA=ON"]
            set_musa_envs()
            global BUILD_TEST_FILE
            BUILD_TEST_FILE = False
        else:
            raise ValueError("Please set --device to a2 or a3 or mlu or cuda or ilu or musa.")

        product = "xllm"
        if self.generate_so:
            product = "libxllm.so"
            cmake_args += ["-DGENERATE_SO=ON"]
        else:
            cmake_args += ["-DGENERATE_SO=OFF"]

        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # check if torch binary is built with cxx11 abi
        if get_cxx_abi():
            cmake_args += ["-DUSE_CXX11_ABI=ON", "-D_GLIBCXX_USE_CXX11_ABI=1"]
        else:
            cmake_args += ["-DUSE_CXX11_ABI=OFF", "-D_GLIBCXX_USE_CXX11_ABI=0"]
        
        build_args = ["--config", build_type]
        build_args += ["-j" + max_jobs]

        env = os.environ.copy()
        env["VCPKG_MAX_CONCURRENCY"] = str(max_jobs)
        print("CMake Args: ", cmake_args)
        print("Env: ", env)

        self.build_cmake_targets(ext, cmake_args, build_args, env, extdir, product)

    def build_cmake_targets(self, ext, cmake_args, build_args, env, extdir, product):
        """Build CMake targets"""
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
            os.path.join(extdir, product),
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

class ExtBuildSingleTest(ExtBuild):
    """Inherit ExtBuild, used to build and run a single test"""
    user_options = ExtBuild.user_options + [
        ("test-name=", None, "name of the test target to build and run"),
    ]

    def initialize_options(self):
        ExtBuild.initialize_options(self)
        self.test_name = None

    def finalize_options(self):
        ExtBuild.finalize_options(self)
        if not self.test_name:
            raise ValueError("--test-name is required for ExtBuildSingleTest")

    def build_cmake_targets(self, ext, cmake_args, build_args, env, extdir, product):
        """Override method: only build the specified test target and run"""
        cmake_dir = get_cmake_dir()
        subprocess.check_call(
            ["cmake", self.base_dir] + cmake_args, cwd=cmake_dir, env=env
        )

        base_build_args = build_args
        # Only build the specified test target
        build_args += ["--target", self.test_name]
        subprocess.check_call(["cmake", "--build", ".", "--verbose"] + build_args, cwd=cmake_dir)

        # Find test executable
        # CMake usually places executables in CMAKE_RUNTIME_OUTPUT_DIRECTORY or build directory
        test_executable = None
        possible_paths = [
            os.path.join(cmake_dir, self.test_name),
            os.path.join(extdir, self.test_name),
            os.path.join(cmake_dir, "xllm", "core", self.test_name),
        ]
        
        # Check possible paths first
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                test_executable = path
                break
        
        # If not found, try recursive search in build directory
        if not test_executable:
            for root, dirs, files in os.walk(cmake_dir):
                if self.test_name in files:
                    candidate = os.path.join(root, self.test_name)
                    if os.access(candidate, os.X_OK):
                        test_executable = candidate
                        break
        
        if not test_executable:
            # If not found, try using ctest to run
            print(f"⚠️  Warning: Could not find test executable {self.test_name}, trying ctest...")
            try:
                subprocess.check_call(
                    ["ctest", "-R", self.test_name, "--verbose"],
                    cwd=cmake_dir,
                    env=env
                )
                print(f"✅ Test {self.test_name} passed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to run test {self.test_name}")
                raise
        else:
            # Run test executable directly
            print(f"🚀 Running test: {test_executable}")
            try:
                subprocess.check_call([test_executable], cwd=os.path.dirname(test_executable), env=env)
                print(f"✅ Test {self.test_name} passed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Test {self.test_name} failed with exit code {e.returncode}")
                raise

class BuildDistWheel(bdist_wheel):
    user_options = bdist_wheel.user_options + [
        ("device=", None, "target device type (a3 or a2 or mlu or cuda or musa)"),
        ("arch=", None, "target arch type (x86 or arm)"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.device = None
        self.arch = None
        # Cache the original dist name early so finalize_options is idempotent
        # and so name changes are visible to egg_info/metadata generation.
        self._base_dist_name = self.distribution.metadata.name

    def finalize_options(self):
        # IMPORTANT: mutate distribution name BEFORE super().finalize_options().
        # bdist_wheel finalization may finalize/cache egg_info metadata; if we
        # change the name afterwards, the wheel filename and METADATA can diverge
        # (pip will reject the wheel as "inconsistent name").
        name = self._base_dist_name

        # generate distribution name suffix
        if self.device:
            name += f"_{self.device}"

        torch_version = get_torch_version(self.device)
        if torch_version:
            name += f"_torch{torch_version}"

        if get_cxx_abi():
            name += "_cxx11_abi"
        else:
            name += "_no_cxx11_abi"

        self.distribution.metadata.name = name
        super().finalize_options()

    def run(self):
        build_ext_cmd = self.get_finalized_command('build_ext')
        build_ext_cmd.device = self.device
        build_ext_cmd.arch = self.arch

        print("🔨 build project...")
        self.run_command('build')

        print("🧪 testing UT...")
        self.run_command('test')

        if self.arch == 'arm':
            ext_path = get_base_dir() + f"/build/lib.linux-aarch64-cpython-{get_python_version()}/"
        else:
            ext_path = get_base_dir() + f"/build/lib.linux-x86_64-cpython-{get_python_version()}/"
        if len(ext_path) == 0:
            print("❌ Build wheel failed, not found path.")
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
                ['ctest', '--parallel', '8', '--repeat', 'until-pass:5'],
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
              print("❌ Testing failed.")
              exit(1)
            return return_code
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            exit(1)

    def run(self):
        self.run_ctest(get_cmake_dir())

class BuildTest(Command):
    """Command to build and run a single test"""
    description = "Build and run a single test target."
    user_options = [
        ("test-name=", None, "name of the test target to build and run"),
        ("device=", None, "target device type (a3 or a2 or mlu or cuda or ilu)"),
        ("arch=", None, "target arch type (x86 or arm)"),
        ("install-xllm-kernels=", None, "install xllm_kernels RPM package (true/false)"),
        ("generate-so=", None, "generate so or binary"),
    ]

    def initialize_options(self):
        self.test_name = None
        self.device = None
        self.arch = None
        self.install_xllm_kernels = None
        self.generate_so = False

    def finalize_options(self):
        if not self.test_name:
            raise ValueError("--test-name is required for build_test command")

    def run(self):
        # Create ExtBuildSingleTest instance and set parameters
        build_ext = ExtBuildSingleTest(self.distribution)
        build_ext.initialize_options()
        build_ext.test_name = self.test_name
        build_ext.device = self.device
        build_ext.arch = self.arch
        build_ext.install_xllm_kernels = self.install_xllm_kernels
        build_ext.generate_so = self.generate_so
        build_ext.finalize_options()
        
        # Ensure extension modules are set
        if not hasattr(build_ext, 'extensions') or not build_ext.extensions:
            build_ext.extensions = self.distribution.ext_modules
        
        # Run build
        build_ext.run()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Setup helper for building xllm',
        epilog='Example: python setup.py build --device a3',
        usage='%(prog)s [COMMAND] [OPTIONS]'
    )
    
    parser.add_argument(
        'setup_args',
        nargs='*',
        metavar='argparse.REMAINDER',
        help='setup command (build, test, bdist_wheel, etc.)'
    )
    
    parser.add_argument(
        '--device',
        type=str.lower,
        choices=['auto', 'a2', 'a3', 'mlu', 'cuda', 'ilu', 'musa'],
        default='auto',
        help='Device type: a2, a3, mlu, ilu, cuda or musa (case-insensitive)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (do not execute pre_build)'
    )
    
    parser.add_argument(
        '--install-xllm-kernels',
        type=str.lower,
        choices=['true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 'on', 'off'],
        default='true',
        help='Whether to install xllm kernels'
    )
    
    parser.add_argument(
        '--generate-so',
        type=str.lower,
        choices=['true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 'on', 'off'],
        default='false',
        help='Whether to generate so or binary'
    )
    
    parser.add_argument(
        '--test-name',
        type=str,
        default=None,
        help='Name of the test target to build and run (for build_test command)'
    )

    args = parser.parse_args()
    
    sys.argv = [sys.argv[0]] + args.setup_args
    
    install_kernels = args.install_xllm_kernels.lower() in ('true', '1', 'yes', 'y', 'on')
    generate_so = args.generate_so.lower() in ('true', '1', 'yes', 'y', 'on')

    return {
        'device': args.device,
        'dry_run': args.dry_run,
        'install_xllm_kernels': install_kernels,
        'generate_so': generate_so,
        'test_name': args.test_name,
    }

if __name__ == "__main__":
    config = parse_arguments()

    arch = get_cpu_arch()
    device = config['device']
    if device == 'auto':
        device = get_device_type()
    print(f"🚀 Build xllm with CPU arch: {arch} and target device: {device}")
    
    if not config['dry_run']:
        pre_build(device)

    install_kernels = config['install_xllm_kernels']
    generate_so = config['generate_so']
    test_name = config.get('test_name')

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
        project_urls={
            "Homepage": "https://xllm.readthedocs.io/zh-cn/latest/",
            "Documentation": "https://xllm.readthedocs.io/zh-cn/latest/",
        },
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Operating System :: POSIX",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        ext_modules=[CMakeExtension("xllm", "xllm/")],
        cmdclass={"build_ext": ExtBuild,
                  "test": TestUT,
                  "build_test": BuildTest,
                  'bdist_wheel': BuildDistWheel},
        options={'build_ext': {
                    'device': device,
                    'arch': arch,
                    'install_xllm_kernels': install_kernels,
                    'generate_so': generate_so
                    },
                 'build_test': {
                    'device': device,
                    'arch': arch,
                    'install_xllm_kernels': install_kernels,
                    'generate_so': generate_so,
                    'test_name': test_name,
                    },
                 'bdist_wheel': {
                    'device': device,
                    'arch': arch,
                    }
                },
        zip_safe=False,
        py_modules=["xllm/launch_xllm", "xllm/__init__",
                    "xllm/pybind/llm", "xllm/pybind/vlm",
                    "xllm/pybind/embedding", "xllm/pybind/util",
                    "xllm/pybind/args"],
        entry_points={
            'console_scripts': [
                'xllm = xllm.launch_xllm:launch_xllm'
            ],
        },
        python_requires=">=3.10",
    )
