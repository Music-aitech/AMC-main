import argparse

from _summary_utils import print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect visual perturbation controls.")
    parser.add_argument("--summary", default="results/table6_visual_perturbation.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print_summary(
        args.summary,
        ["condition", "CLAP_similarity_mean", "CLAP_similarity_sd", "FAD", "KL", "tempo_error_percent"],
    )
    print("Visual perturbation summary inspection complete.")


if __name__ == "__main__":
    main()
