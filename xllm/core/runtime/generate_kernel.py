#!/usr/bin/env python3
"""
生成一个简单的 matmul kernel 并导出 TVM FFI Runtime Module
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys

# 设置 PYTHONPATH 以包含 tilelang
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# TILELANG_DIR = os.path.join(SCRIPT_DIR, "..", "tilelang")
# if os.path.exists(TILELANG_DIR):
#     sys.path.insert(0, TILELANG_DIR)


import tilelang
import tilelang.language as T
# import tvm
# 确保输出目录存在
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "kernels")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"输出目录: {OUTPUT_DIR}")

# 定义 matmul kernel
@tilelang.jit(execution_backend="tvm_ffi", target="cuda")
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def simple_matmul(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return simple_matmul


# 配置 kernel 参数
M = 1024
N = 1024
K = 512
block_M = 128
block_N = 128
block_K = 32

print(f"编译 matmul kernel: M={M}, N={N}, K={K}")
print(f"Block 大小: block_M={block_M}, block_N={block_N}, block_K={block_K}")

# 编译 kernel
kernel = matmul(M, N, K, block_M, block_N, block_K)

# 获取 adapter
adapter = kernel.adapter

# 添加调试信息
print("\n=== Adapter 调试信息 ===")
print(f"Adapter 类型: {type(adapter)}")
print(f"Adapter 是否有 rt_mod 属性: {hasattr(adapter, 'rt_mod')}")

# # 尝试多种方法获取 rt_mod
rt_mod = None

# 方法1: 直接访问 adapter.rt_mod
if hasattr(adapter, 'rt_mod'):
    rt_mod = adapter.rt_mod
    print(f"方法1成功: adapter.rt_mod = {rt_mod} (类型: {type(rt_mod)})")



if rt_mod is None:
    print("\n错误: 无法获取 rt_mod")
    print("可能的原因:")
    print("  1. tvm_ffi backend 需要显式构建 runtime module")
    print("  2. 编译过程未完成或失败")
    print("  3. adapter 的 API 与预期不同")
    print("\n请检查:")
    print("  - tilelang 版本是否支持 tvm_ffi backend")
    print("  - 是否需要额外的构建步骤")
    sys.exit(1)

# 导出 rt_mod（已经在上面的代码中获取）
rt_mod_path = os.path.join(OUTPUT_DIR, "simple_matmul.so")

print(f"导出 rt_mod 到: {rt_mod_path}")
rt_mod.export_library(rt_mod_path)

# 保存元信息
import json
meta_info = {
    "kernel_name": "simple_matmul",
    "global_symbol": "simple_matmul",
    "ffi_symbol": "tvm_ffi_simple_matmul",
    "M": M,
    "N": N,
    "K": K,
    "num_inputs": 2,
    "num_outputs": 1,
    "device_type": 2,  # kDLCUDA
    "device_id": 0,
    "params": [
        {"name": "A", "shape": [M, K], "dtype": "float16"},
        {"name": "B", "shape": [K, N], "dtype": "float16"},
        {"name": "C", "shape": [M, N], "dtype": "float16"},
    ]
}

meta_path = os.path.join(OUTPUT_DIR, "simple_matmul_meta.json")
with open(meta_path, 'w') as f:
    json.dump(meta_info, f, indent=2)

print(f"导出元信息到: {meta_path}")

# 打印一些调试信息
print("\n=== Kernel 信息 ===")
print(f"PrimFunc 名称: {adapter.prim_func.attrs.get('global_symbol', 'unknown')}")
print(f"Target: {adapter.target}")

# 打印设备源码（可选）
try:
    device_source = adapter.get_device_source()
    device_source_path = os.path.join(OUTPUT_DIR, "simple_matmul_device.cu")
    with open(device_source_path, 'w') as f:
        f.write(device_source)
    print(f"设备源码已保存到: {device_source_path}")
except Exception as e:
    print(f"无法保存设备源码: {e}")

# 打印 host 源码（可选）
try:
    host_source = adapter.get_host_source()
    host_source_path = os.path.join(OUTPUT_DIR, "simple_matmul_host.cpp")
    with open(host_source_path, 'w') as f:
        f.write(host_source)
    print(f"Host 源码已保存到: {host_source_path}")
except Exception as e:
    print(f"无法保存 host 源码: {e}")

print("\n=== 生成完成 ===")
print(f"请使用以下文件进行 C++ 集成:")
print(f"  - Runtime Module: {rt_mod_path}")
print(f"  - 元信息: {meta_path}")


# def run_python_runtime_smoke(so_path: str) -> None:
#     print("\n=== Python 侧 load + run smoke ===")
#     import numpy as np
    
#     import tvm_ffi

#     print(f'tilelang.__file__: {tilelang.__file__}')
#     print(f'tvm-ffi.__file__: {tvm_ffi.__file__}')
#     print(f'tvm.__file__: {tilelang.tvm.__file__}')
    

#     if hasattr(tilelang.tvm, "cuda"):
#         dev = tilelang.tvm.cuda(0)
#     else:
#         dev = tilelang.tvm.device("cuda", 0)
#     if not dev.exist:
#         raise RuntimeError("tvm.cuda(0) 不可用，无法做 Python 侧 smoke。")

#     mod = tilelang.tvm.runtime.load_module(so_path)
#     func = None
#     resolved_symbol = None
#     for symbol in ("main", "__tvm_ffi_main", "simple_matmul", "tvm_ffi_simple_matmul"):
#         try:
#             func = mod[symbol]
#             resolved_symbol = symbol
#             break
#         except Exception:
#             continue

#     if func is None:
#         raise RuntimeError("在 Python load_module 后未找到可调用入口符号。")
#     print(f"Python 侧解析到符号: {resolved_symbol}")

#     import torch

#     torch.manual_seed(0)
#     device = torch.device("cuda:0")
#     dtype = torch.float16
#     A = torch.randn((M, K), device=device, dtype=dtype)
#     B = torch.randn((K, N), device=device, dtype=dtype)
#     C = torch.zeros((M, N), device=device, dtype=dtype)

#     # Prefer directly passing torch.Tensor. In tvm-ffi pipelines this is often
#     # accepted via DLPack bridge and avoids relying on tvm.nd namespace variants.
#     func(A, B, C)
#     if hasattr(dev, "sync"):
#         dev.sync()
#     else:
#         torch.cuda.synchronize()
#     c_out = C.detach().cpu().numpy()
#     a_np = A.detach().cpu().numpy()
#     b_np = B.detach().cpu().numpy()

#     sub_m = min(32, M)
#     sub_n = min(32, N)
#     c_ref_sub = (
#         a_np[:sub_m, :].astype(np.float32) @ b_np[:, :sub_n].astype(np.float32)
#     ).astype(np.float16)
#     c_out_sub = c_out[:sub_m, :sub_n]
#     ok = np.allclose(c_out_sub, c_ref_sub, rtol=1e-1, atol=1e-1)
#     max_abs_diff = float(np.max(np.abs(c_out_sub.astype(np.float32) - c_ref_sub.astype(np.float32))))
#     print(f"Python smoke allclose={ok}, max_abs_diff={max_abs_diff:.6f}")
#     if not ok:
#         raise RuntimeError("Python 侧 smoke 校验失败：TileLang matmul 结果与参考不一致。")
#     print("Python 侧 smoke 通过。")


# if os.environ.get("TL_RUN_PY_SMOKE", "1") != "0":
#     run_python_runtime_smoke(rt_mod_path)
