import argparse
from pathlib import Path

from _summary_utils import print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect AMC ablation configs and results.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/ablation"))
    parser.add_argument("--summary", default="results/table4_ablation_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = sorted(args.config_dir.glob("*.yaml"))
    if not configs:
        raise SystemExit(f"No ablation configs found in {args.config_dir}")
    print("Available ablation configurations:")
    for config in configs:
        print(f"- {config}")
    print_summary(args.summary, ["variant", "FAD", "KL"])
    print("Ablation summary inspection complete.")


if __name__ == "__main__":
    main()
