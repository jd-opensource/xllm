import argparse
import json
import os
import time
from typing import Any, Dict, Optional

try:
    import acl 
    ret = acl.rt.set_device(0)
    if ret not in (None, 0):
        print(f"[WARN] acl.rt.set_device(0) returned {ret}.")
except ImportError:
    print(f"[WARN] protocol={protocol} import acl failed.")
except Exception as exc:  # pylint: disable=broad-except
    print(f"[WARN] acl.rt.set_device(0) failed: {exc}")

from mooncake.store import MooncakeDistributedStore

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start Mooncake distributed object store provider."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        raise ValueError("Config path is required")

    with open(config_path, "r", encoding="utf-8") as f:
        file_config = json.load(f)

    if not isinstance(file_config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")

    # Accept common typo alias: "protocal".
    if "protocal" in file_config and "protocol" not in file_config:
        file_config["protocol"] = file_config["protocal"]

    required_fields = [
        "local_hostname",
        "metadata_server",
        "global_segment_size",
        "local_buffer_size",
        "protocol",
        "master_server_address",
    ]
    missing = [field for field in required_fields if field not in file_config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    file_config["global_segment_size"] = int(file_config["global_segment_size"])
    file_config["local_buffer_size"] = int(file_config["local_buffer_size"])
    file_config.setdefault("device_name", "")
    file_config.setdefault("ub_protocol_env", {})

    if not isinstance(file_config["ub_protocol_env"], dict):
        raise ValueError("'ub_protocol_env' must be a JSON object")

    print(f"[INFO] Loaded config: {config_path}")
    return file_config


def _to_env_value(value: Any) -> str:
    # UB side parses several vars with atoi(), so bool should map to 1/0.
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def apply_ub_protocol_env(config: Dict[str, Any]) -> None:
    if str(config["protocol"]).lower() != "ub":
        return

    ub_env = config.get("ub_protocol_env", {})
    for key, value in ub_env.items():
        os.environ[str(key)] = _to_env_value(value)

    if ub_env:
        print(f"[INFO] Applied {len(ub_env)} env vars from ub_protocol_env.")


def start_provider(config_path: Optional[str] = None) -> None:
    config = load_config(config_path)
    apply_ub_protocol_env(config)

    store = MooncakeDistributedStore()
    retcode = store.setup(
        config["local_hostname"],
        config["metadata_server"],
        config["global_segment_size"],
        config["local_buffer_size"],
        config["protocol"],
        config["device_name"],
        config["master_server_address"],
    )
    if retcode:
        raise SystemExit(1)

    while True:
        time.sleep(100)  # Keep provider process alive.


# Keep backward compatibility with existing callers.
def startProvider(config_path: Optional[str] = None) -> None:  # noqa: N802
    start_provider(config_path)


if __name__ == "__main__":
    args = parse_args()
    start_provider(args.config)
