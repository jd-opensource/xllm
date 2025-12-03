import importlib.util
import os
import xllm
import sys

def get_python_version():
    return f"{sys.version_info.major}{sys.version_info.minor}"

install_path_x86 = os.path.dirname(xllm.__file__) + f"/xllm_export.cpython-{get_python_version()}-x86_64-linux-gnu.so"
install_path_arm = os.path.dirname(xllm.__file__) + f"/xllm_export.cpython-{get_python_version()}-aarch64-linux-gnu.so"
if os.path.exists(install_path_x86):
    install_path = install_path_x86
elif os.path.exists(install_path_arm):
    install_path = install_path_arm
else:
    raise ValueError("cannot open shared object file: No such file or directory, required ", install_path_x86, " or ", install_path_arm)
export_so_path = os.path.abspath(install_path)
spec = importlib.util.spec_from_file_location("xllm_export", export_so_path)
xllm_export = importlib.util.module_from_spec(spec)

from xllm.pybind.embedding import Embedding
from xllm.pybind.llm import LLM
from xllm.pybind.vlm import VLM
from xllm.pybind.args import ArgumentParser
from xllm_export import (LLMMaster, Options, RequestParams, RequestOutput,
                         SequenceOutput, Status, StatusCode, MMType, MMData)

__all__ = [
    "ArgumentParser",
    "Embedding",
    "LLM",
    "LLMMaster",
    "VLM",
    "VLMMaster"
    "Options",
    "RequestParams",
    "RequestOutput",
    "SequenceOutput",
    "Status",
    "StatusCode",
]
