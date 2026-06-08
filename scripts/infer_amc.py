import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import AMCModel


def parse_args():
    parser = argparse.ArgumentParser(description="AMC inference interface.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_amc.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--visual-features", type=Path)
    parser.add_argument("--acoustic-features", type=Path)
    parser.add_argument("--text-features", type=Path)
    parser.add_argument("--output", type=Path, help="Output waveform path.")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_model(config):
    return AMCModel(
        v_dim=config["visual_dim"],
        a_dim=config["acoustic_dim"],
        t_dim=config["text_dim"],
        hid_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        amf_temperature=config["amf_temperature"],
        block_size=config["block_size"],
        num_cmt_blocks=config["num_cmt_blocks"],
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config).eval()

    if args.smoke_test:
        length = config.get("smoke_sequence_length", 4)
        with torch.no_grad():
            latent, global_music = model(
                torch.randn(1, length, config["visual_dim"]),
                torch.randn(1, length, config["acoustic_dim"]),
                torch.randn(1, length, config["text_dim"]),
            )
        print(f"AMC inference smoke test passed: latent={tuple(latent.shape)}, global={tuple(global_music.shape)}")
        print("Synthetic latent-path validation complete.")
        return

    required = [
        args.checkpoint,
        args.visual_features,
        args.acoustic_features,
        args.text_features,
        args.output,
    ]
    if not all(required):
        raise SystemExit(
            "Real inference requires --checkpoint, all three feature paths, and "
            "--output. Use --smoke-test to inspect the released latent path."
        )
    raise SystemExit("Waveform decoding requires project-specific generation backend configuration.")


if __name__ == "__main__":
    main()
