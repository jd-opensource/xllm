# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triton AOT compilation script for xllm MLU kernels.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# Import kernel definitions from separate file
from kernel_config import (
    KernelConfig,
    KernelSignature,
    get_kernel_configs,
    DEFAULT_ARCHS,
    mlu_ops_path,
)


def _compile_single_signature(
    idx: int,
    sig: KernelSignature,
    kernel_key: str,
    config: KernelConfig,
    kernel_file: Path,
    lib_dir: Path,
    include_dir: Path,
    archs: List[str],
    verbose: bool = False,
) -> Tuple[Path, Path]:
    """Compile a single signature variant of a kernel."""
    device_kernel_name = f"{config.device_kernel_name}_{idx}"
    output_name = f"{config.device_kernel_name}_cc_{idx}"
    header_path = include_dir / f"{output_name}.h"

    cmd = [
        sys.executable, "-m", "triton.tools.mlu_compile",
        "--kernel-name", config.full_kernel_name,
        "--device-kernel-name", device_kernel_name,
        "--signature", sig.params,
        "--archs", *archs,
        "--type", "obj",
        "--out-name", output_name,
        str(kernel_file),
    ]

    if verbose:
        print(f"  Compiling variant {idx} ({sig.name}): {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(lib_dir))
    if result.returncode != 0:
        print(f"Error compiling {kernel_key} variant {idx} ({sig.name}):")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"Failed to compile {kernel_key} variant {idx}")

    generated_o = lib_dir / f"{output_name}.o"
    generated_h = lib_dir / f"{output_name}.h"

    if generated_h.exists():
        generated_h.rename(header_path)

    if verbose:
        print(f"    Generated: {generated_o}")

    return generated_o, header_path


def compile_kernel(
    kernel_key: str,
    config: KernelConfig,
    script_dir: Path,
    output_dir: Path,
    archs: List[str],
    verbose: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Compile a single kernel with multiple signatures in parallel."""
    kernel_file = Path(config.kernel_file)
    if not kernel_file.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

    lib_dir = output_dir / "lib"
    include_dir = output_dir / "include"
    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    header_files = []

    with ThreadPoolExecutor(max_workers=len(config.signatures)) as executor:
        futures = {
            executor.submit(
                _compile_single_signature,
                idx, sig, kernel_key, config, kernel_file,
                lib_dir, include_dir, archs, verbose
            ): idx
            for idx, sig in enumerate(config.signatures)
        }

        results = {}
        for future in as_completed(futures):
            try:
                idx = futures[future]
                generated_o, header_path = future.result()
                results[idx] = (generated_o, header_path)
            except Exception as e:
                idx = futures[future]
                sig = config.signatures[idx]
                print(f"Error compiling {kernel_key} variant {idx} ({sig.name}): {e}")
                raise

    for idx in sorted(results.keys()):
        generated_o, header_path = results[idx]
        if generated_o.exists():
            output_files.append(generated_o)
        header_files.append(header_path)

    return output_files, header_files


def link_kernels(
    kernel_key: str,
    config: KernelConfig,
    header_files: List[Path],
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """Link multiple kernel header files into a single entry-point."""
    if len(header_files) == 0:
        return

    include_dir = output_dir / "include"
    output_name = include_dir / config.device_kernel_name

    cmd = [
        sys.executable, "-m", "triton.tools.mlu_link",
        *[str(h) for h in header_files],
        "-o", str(output_name),
    ]

    if verbose:
        print(f"  Linking headers: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error linking {kernel_key}:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError(f"Failed to link {kernel_key}")

    if verbose:
        print(f"    Linked to: {output_name}.h and {output_name}.c")

    for header_file in header_files:
        if header_file.exists():
            header_file.unlink()
            if verbose:
                print(f"    Removed: {header_file}")


def generate_cmake_config(output_dir: Path, verbose: bool = False):
    """Generate CMake configuration file with expected kernel object files."""
    lib_dir = output_dir / "lib"
    include_dir = output_dir / "include"
    cmake_file = lib_dir / "kernel_objects.cmake"
    
    # Calculate expected object files based on registered kernels
    object_files = []
    gen_c_files = []
    for kernel_key, config in get_kernel_configs().items():
        num_signatures = len(config.signatures)
        for idx in range(num_signatures):
            output_name = f"{config.device_kernel_name}_cc_{idx}"
            obj_file = lib_dir / f"{output_name}.o"
            object_files.append(str(obj_file))
        
        # Add generated .c file for this kernel (from link_kernels)
        gen_c_file = include_dir / f"{config.device_kernel_name}.c"
        gen_c_files.append(str(gen_c_file))
    
    # Write CMake config file
    with open(cmake_file, "w") as f:
        f.write("# Auto-generated by compile_kernel.py\n")
        f.write("# Do not edit manually\n")
        f.write("set(TRITON_KERNEL_OBJECT_FILES\n")
        for obj_file in object_files:
            f.write(f"    {obj_file}\n")
        f.write(")\n")
        f.write("\n")
        f.write("# Generated .c files from link_kernels\n")
        f.write("set(TRITON_KERNEL_GEN_C_FILES\n")
        for gen_c_file in gen_c_files:
            f.write(f"    {gen_c_file}\n")
        f.write(")\n")
    
    if verbose:
        print(f"Generated CMake config: {cmake_file}")
        print(f"Expected {len(object_files)} object files")
        print(f"Expected {len(gen_c_files)} generated .c files")
    
    return object_files


def main():
    parser = argparse.ArgumentParser(
        description="Compile Triton kernels for xllm MLU backend"
    )
    parser.add_argument(
        "--kernels", "-k",
        nargs="+",
        default=list(get_kernel_configs().keys()),
        help="Kernels to compile"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for compiled kernels"
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=DEFAULT_ARCHS,
        help="Target architectures"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--generate-cmake-only",
        action="store_true",
        help="Only generate CMake config without compiling"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / "kernel"

    # Ensure output directories exist
    lib_dir = output_dir / "lib"
    include_dir = output_dir / "include"
    lib_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)

    # Always generate CMake config first (even for compilation)
    generate_cmake_config(output_dir, args.verbose)

    if args.generate_cmake_only:
        print("CMake config generated. Exiting without compilation.")
        return

    print("=== Triton Kernel Compilation for xllm MLU ===")
    print(f"Script dir: {script_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target architectures: {args.archs}")
    print()

    compiled = {}
    for kernel_key in args.kernels:
        if kernel_key not in get_kernel_configs():
            print(f"Warning: Unknown kernel '{kernel_key}', skipping")
            continue

        config = get_kernel_configs()[kernel_key]
        print(f"Compiling {kernel_key}...")
        print(f"  File: {config.kernel_file}")
        print(f"  Variants: {', '.join(sig.name for sig in config.signatures)}")

        try:
            files, headers = compile_kernel(
                kernel_key,
                config,
                script_dir,
                output_dir,
                args.archs,
                args.verbose,
            )
            compiled[kernel_key] = files
            print(f"  Generated {len(files)} variant(s)")

            if len(headers) > 0:
                print(f"  Linking {len(headers)} header files...")
                link_kernels(kernel_key, config, headers, output_dir, args.verbose)
        except Exception as e:
            print(f"  Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\n=== Compilation Summary ===")
    total_files = sum(len(f) for f in compiled.values())
    print(f"Total kernels compiled: {len(compiled)}")
    print(f"Total object files: {total_files}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
