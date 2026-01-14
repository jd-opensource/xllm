import os
import platform


def get_cxx_abi():
    try:
        import torch
        return torch.compiled_with_cxx11_abi()
    except ImportError:
        return False

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

def get_ixformer_root_path():
    try:
        import ixformer
        import os
        return os.path.dirname(os.path.abspath(ixformer.__file__))
    except ImportError:
        return None
    
def get_cuda_root_path():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError(
                "PyTorch was not built with CUDA, or nvcc is not in PATH. "
                "Please set CUDA_TOOLKIT_ROOT_DIR manually."
            )
        return CUDA_HOME
    except ImportError:
        return None
    
def set_common_envs():
    os.environ["PYTHON_INCLUDE_PATH"] = get_python_include_path()
    os.environ["PYTHON_LIB_PATH"] =  get_torch_root_path()
    os.environ["LIBTORCH_ROOT"] = get_torch_root_path()
    os.environ["PYTORCH_INSTALL_PATH"] = get_torch_root_path()

def set_npu_envs():
    PYTORCH_NPU_INSTALL_PATH = os.getenv("PYTORCH_NPU_INSTALL_PATH")
    if not PYTORCH_NPU_INSTALL_PATH:
        os.environ["PYTORCH_NPU_INSTALL_PATH"] = "/usr/local/libtorch_npu"

    XLLM_KERNELS_PATH = os.getenv("XLLM_KERNELS_PATH")
    if not XLLM_KERNELS_PATH:
        os.environ["XLLM_KERNELS_PATH"] = "/usr/local/xllm_kernels"

    set_common_envs()
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
        NPU_TOOLKIT_HOME+"/opp/vendors/xllm/op_api/lib" + ":" + \
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
    set_common_envs()
    os.environ["PYTORCH_MLU_INSTALL_PATH"] = get_torch_mlu_root_path()
    
def set_cuda_envs():
    set_common_envs()
    os.environ["CUDA_TOOLKIT_ROOT_DIR"] = get_cuda_root_path()

def set_ilu_envs():
    set_common_envs()
    os.environ["IXFORMER_INSTALL_PATH"] = get_ixformer_root_path()
