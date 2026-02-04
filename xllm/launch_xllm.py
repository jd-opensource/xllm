import os
import platform
import subprocess
import sys
from typing import Dict

import xllm


def launch_xllm() -> int:
    system = platform.system()
    binary_name: str = {
        "Linux": "xllm",
        # "Windows"
        # "Darwin"
    }.get(system, "xllm")

    bin_path: str = os.path.dirname(xllm.__file__) + "/xllm"

    result = subprocess.run([str(bin_path)] + sys.argv[1:])
    return result.returncode
