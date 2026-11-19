"""Point every simulation config under data/ at a different circuit.

Rewrites the top-level "network" value in each data/*/sim_with_stim.json and
data/*/sim_with_stim_new.json, leaving the rest of every file byte-identical so
the diff is one line per config.

Nothing is written unless the given circuit config exists and every simulation
config checks out, so a wrong path cannot leave the configs half-rewritten.

Usage:
    python -m src.set_network_path /path/to/O1_data/circuit_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


CONFIG_NAMES = ("sim_with_stim.json", "sim_with_stim_new.json")

# The one "network": "..." entry, keeping its value's JSON escapes intact.
NETWORK_RE = re.compile(r'("network"\s*:\s*)"(?:[^"\\]|\\.)*"')


def find_configs(data_dir: Path, filenames: list[str]) -> list[Path]:
    """Simulation configs one level below data_dir, skipping dotted folders."""
    folders = sorted(
        path for path in data_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    return [
        folder / name
        for folder in folders
        for name in filenames
        if (folder / name).is_file()
    ]


def resolve_target(config: Path, network: str) -> Path:
    """Where network points from config, the way neurodamus resolves it."""
    target = Path(network)
    return target if target.is_absolute() else config.parent / target


def check_config(config: Path, network: str) -> list[str]:
    """Reasons this config cannot be rewritten, as messages."""
    text = config.read_text()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        return [f"{config}: not valid JSON ({err})"]

    problems = []
    if "network" not in parsed:
        problems.append(f'{config}: no top-level "network" key')
    found = len(NETWORK_RE.findall(text))
    if found != 1:
        problems.append(f'{config}: found {found} "network" entries, expected exactly 1')
    return problems


def set_network(config: Path, network: str) -> bool:
    """Rewrite the network value in place. True if the file changed."""
    text = config.read_text()
    # json.dumps, not a plain string, so backslashes and quotes survive both the
    # substitution and the JSON that comes out of it.
    new_text = NETWORK_RE.sub(lambda m: m.group(1) + json.dumps(network), text, count=1)
    if new_text == text:
        return False

    config.write_text(new_text)
    written = json.loads(new_text)["network"]
    if written != network:
        raise RuntimeError(
            f"{config}: network is {written!r} after rewriting, expected {network!r}"
        )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("network", help="circuit_config.json the simulations should use")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--filenames", nargs="+", default=list(CONFIG_NAMES))
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"ERROR: {args.data_dir} is not a directory")

    configs = find_configs(args.data_dir, args.filenames)
    if not configs:
        sys.exit(
            f"ERROR: no {' or '.join(args.filenames)} under {args.data_dir}/*/"
        )

    problems = []
    missing = sorted(
        {
            str(resolve_target(config, args.network))
            for config in configs
            if not resolve_target(config, args.network).is_file()
        }
    )
    if missing:
        # A relative path misses from every folder at once, so list a few rather
        # than one line per config.
        for target in missing[:3]:
            problems.append(f"circuit config not found: {target}")
        if len(missing) > 3:
            problems.append(f"...and {len(missing) - 3} more paths that do not exist")
        if not Path(args.network).is_absolute():
            problems.append(
                f"{args.network!r} is relative, so it is resolved from each config's "
                "own folder; pass an absolute path instead"
            )
    for config in configs:
        problems.extend(check_config(config, args.network))

    if problems:
        print(f"Refusing to change anything, {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(1)

    width = max(len(str(config)) for config in configs)
    updated = 0
    for config in configs:
        changed = set_network(config, args.network)
        updated += changed
        print(f"  {str(config):<{width}}  {'updated' if changed else 'unchanged'}")
    print(f"{updated} updated, {len(configs) - updated} unchanged")
