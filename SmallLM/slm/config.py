"""Training settings: built-in defaults < a JSON config file < command-line flags.

    python train.py --config configs/tiny.json --max-iters 3000

loads configs/tiny.json, then overrides max_iters. Settings are plain
attributes afterwards: `settings.max_iters`.
"""

import argparse
import json
from types import SimpleNamespace


def add_flags(parser, defaults):
    """Add a --flag for every default. Flags default to None so we can tell
    "not given" apart from "given the default value"."""
    for name, default in defaults.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif default is None:
            parser.add_argument(flag, default=None)
        else:
            parser.add_argument(flag, type=type(default), default=None)


def load_settings(defaults, description=None):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON file of settings")
    add_flags(parser, defaults)
    args = parser.parse_args()

    settings = dict(defaults)
    if args.config:
        with open(args.config) as f:
            from_file = json.load(f)
        unknown = set(from_file) - set(defaults)
        if unknown:
            raise ValueError(f"unknown settings in {args.config}: {sorted(unknown)}")
        settings.update(from_file)
    settings.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    return SimpleNamespace(**settings)
