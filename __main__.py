from __future__ import annotations

import argparse
import logging
import sys

from config import Settings
from bridge_wind_fragility import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge wind fragility pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-fsi", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)

    settings = Settings.from_yaml(args.config)
    run(settings, use_fsi=not args.no_fsi)


if __name__ == "__main__":
    main()
