"""
hw_config.py — Hardware config advisor CLI.

Usage:
    uv run python hw_config.py              # print recommended config
    uv run python hw_config.py --export     # also write .env

Logic lives in tinydet/hw.py.
"""
import argparse
from tinydet.hw import detect_hardware, recommend_config, print_report, export_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware config advisor")
    parser.add_argument("--export", action="store_true",
                        help="Merge recommended values into .env (preserves existing keys)")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    hw = detect_hardware()
    cfg = recommend_config(hw)
    print_report(hw, cfg)

    if args.export:
        export_env(cfg, args.env_file)


if __name__ == "__main__":
    main()
