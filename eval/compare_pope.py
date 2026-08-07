import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.native) as input_file:
        native = json.load(input_file)
    with open(args.method) as input_file:
        method = json.load(input_file)

    metrics = ("accuracy", "precision", "recall", "f1")
    comparison = {
        "native": native,
        "visual_evidence": method,
        "delta": {
            metric: method[metric] - native[metric]
            for metric in metrics
        },
        "count_delta": {
            key: method[key] - native[key]
            for key in ("TP", "TN", "FP", "FN", "oom")
        },
    }
    serialized = json.dumps(comparison, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)


if __name__ == "__main__":
    main()
