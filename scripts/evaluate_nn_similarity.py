import argparse

from _summary_utils import print_summary, stop_before_recomputation


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect or recompute NN similarity.")
    parser.add_argument("--summary", help="Manuscript summary CSV to print.")
    parser.add_argument("--predictions-dir", help="Directory of generated embeddings.")
    parser.add_argument("--references-dir", help="Directory of corpus embeddings.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.summary:
        print_summary(
            args.summary,
            ["method", "train_nn_similarity", "test_nn_similarity", "FAD"],
        )
        return
    stop_before_recomputation(args.predictions_dir, args.references_dir, "NN similarity")


if __name__ == "__main__":
    main()
