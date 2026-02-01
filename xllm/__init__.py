import importlib.util
import os
import sys
import sysconfig


def _get_python_version_tag() -> str:
    # returns "310", "311", ...
    return sysconfig.get_python_version().replace(".", "")


def _find_export_so_path() -> str:
    pkg_dir = os.path.dirname(__file__)
    pyver = _get_python_version_tag()

    # Preferred, exact tags we build for today.
    candidates = [
        os.path.join(pkg_dir, f"xllm_export.cpython-{pyver}-x86_64-linux-gnu.so"),
        os.path.join(pkg_dir, f"xllm_export.cpython-{pyver}-aarch64-linux-gnu.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # Fallback: accept any xllm_export*.so that got packaged (tag may differ).
    for fname in os.listdir(pkg_dir):
        if fname.startswith("xllm_export") and fname.endswith(".so"):
            return os.path.abspath(os.path.join(pkg_dir, fname))

    raise ImportError(
        f"cannot find xllm_export shared library under {pkg_dir!r}. "
        f"Expected one of: {candidates!r}"
    )


_export_so_path = _find_export_so_path()
_spec = importlib.util.spec_from_file_location("xllm_export", _export_so_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"failed to create import spec for xllm_export: {_export_so_path}")

# Make `import xllm_export` work for submodules (pybind/*) by loading and
# registering it before importing any modules that depend on it.
xllm_export = importlib.util.module_from_spec(_spec)
sys.modules["xllm_export"] = xllm_export
_spec.loader.exec_module(xllm_export)

from xllm.pybind.embedding import Embedding
from xllm.pybind.llm import LLM
from xllm.pybind.vlm import VLM
from xllm.pybind.args import ArgumentParser
from xllm_export import (
    LLMMaster,
    VLMMaster,
    Options,
    RequestParams,
    RequestOutput,
    SequenceOutput,
    Status,
    StatusCode,
    MMType,
    MMData,
)

__all__ = [
    "ArgumentParser",
    "Embedding",
    "LLM",
    "LLMMaster",
    "VLM",
    "VLMMaster",
    "Options",
    "RequestParams",
    "RequestOutput",
    "SequenceOutput",
    "Status",
    "StatusCode",
]
