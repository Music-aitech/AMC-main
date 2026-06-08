import argparse

from _summary_utils import print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect missing-modality controls.")
    parser.add_argument("--summary", default="results/table7_missing_modality.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    print_summary(
        args.summary,
        ["modality_setting", "FAD", "KL", "tempo_error_percent", "CLAP_similarity"],
    )
    print("Missing-modality summary inspection complete.")


if __name__ == "__main__":
    main()
