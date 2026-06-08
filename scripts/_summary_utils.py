import csv
from pathlib import Path


def print_summary(path, columns):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Summary CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise SystemExit(f"Summary CSV is empty: {path}")

    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise SystemExit(f"Missing expected columns in {path}: {', '.join(missing)}")

    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("Summary mode: manuscript-reported aggregate values; no metrics recomputed.")
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(str(row[column]).ljust(widths[column]) for column in columns))


def stop_before_recomputation(predictions_dir, references_dir, metric_name):
    if not predictions_dir or not references_dir:
        raise SystemExit(
            f"{metric_name} recomputation requires both --predictions-dir and "
            "--references-dir. Use --summary to inspect reported values."
        )
    predictions_dir = Path(predictions_dir)
    references_dir = Path(references_dir)
    if not predictions_dir.is_dir() or not references_dir.is_dir():
        raise SystemExit(
            f"Real metric inputs not found: predictions={predictions_dir}, "
            f"references={references_dir}"
        )
    raise SystemExit(
        f"{metric_name} raw-data execution requires project-specific metric backend configuration."
    )
