import os
import platform
import subprocess
import sys
import xllm


def launch_xllm():
    system = platform.system()
    binary_name = {
        "Linux": "xllm",
        # "Windows"
        # "Darwin"
    }.get(system, "xllm")
    
    bin_path = os.path.dirname(xllm.__file__) + "/xllm"

    result = subprocess.run([str(bin_path)] + sys.argv[1:])
    return result.returncode
