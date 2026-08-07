"""Compare aligned one-block Native and visual-margin benchmark results."""

import argparse
import json
from pathlib import Path


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmmu_native", required=True)
    parser.add_argument("--mmmu_margin", required=True)
    parser.add_argument("--hallusion_native", required=True)
    parser.add_argument("--hallusion_margin", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mmmu_native = load(args.mmmu_native)["overall"]
    mmmu_margin = load(args.mmmu_margin)["overall"]
    hallusion_native = load(args.hallusion_native)["metrics"]
    hallusion_margin = load(args.hallusion_margin)["metrics"]

    comparison = {
        "setting": {
            "gen_length": 256,
            "block_length": 256,
            "steps": 32,
            "tokens_per_step": 8,
        },
        "MMMU": {
            "Native": mmmu_native,
            "VisualMargin": mmmu_margin,
            "accuracy_delta": mmmu_margin["accuracy"] - mmmu_native["accuracy"],
        },
        "HallusionBench": {},
    }
    for metric in ["aAcc", "qAcc", "fAcc"]:
        native = hallusion_native[metric]
        margin = hallusion_margin[metric]
        comparison["HallusionBench"][metric] = {
            "Native": native,
            "VisualMargin": margin,
            "accuracy_delta": margin["accuracy"] - native["accuracy"],
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
