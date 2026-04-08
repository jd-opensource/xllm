from __future__ import annotations

import argparse

from ..bootstrap import PREPARE_ASCEND_COMMAND, prepare_ascend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare third_party/tilelang-ascend for xLLM Ascend TileLang builds."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerunning install_ascend.sh even when cached artifacts look ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tilelang_root = prepare_ascend(force=args.force)
    print(f"[INFO] tilelang-ascend is ready under: {tilelang_root}")
    print("[INFO] Next step: run your usual `python setup.py build ...` or `python setup.py test ...` command.")
    print(f"[INFO] Re-run this step explicitly with: `{PREPARE_ASCEND_COMMAND}`")


if __name__ == "__main__":
    main()
