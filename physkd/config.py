"""YAML configuration files as argparse defaults: values on the command line override the file."""
import argparse
import sys

import yaml


def parse(ap, required=()):
    ap.add_argument("--config", default=None, help="YAML file with default values for the options below")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    if known.config:
        with open(known.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        dests = {a.dest for a in ap._actions}
        unknown = sorted(set(cfg) - dests)
        if unknown:
            sys.exit(f"{known.config}: unknown option(s) {unknown}")
        ap.set_defaults(**cfg)
    a = ap.parse_args()
    missing = [r for r in required if getattr(a, r, None) in (None, [], "")]
    if missing:
        ap.error("missing required option(s): " + ", ".join("--" + m for m in missing))
    return a
