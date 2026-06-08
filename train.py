import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model import AMCModel


def contrastive_loss(g_music, g_text, temp=0.1):
    g_m = F.normalize(g_music, p=2, dim=-1)
    g_t = F.normalize(g_text, p=2, dim=-1)

    logits = torch.matmul(g_m, g_t.T) / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def load_config(path):
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def build_model(config):
    return AMCModel(
        v_dim=config.get("visual_dim", 1024),
        a_dim=config.get("acoustic_dim", 512),
        t_dim=config.get("text_dim", 768),
        hid_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        amf_temperature=config["amf_temperature"],
        block_size=config["block_size"],
        num_cmt_blocks=config.get("num_cmt_blocks", 4),
    )


def run_smoke_test(config):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    batch_size = config.get("smoke_batch_size", 2)
    sequence_length = config.get("smoke_sequence_length", 4)
    v = torch.randn(batch_size, sequence_length, config.get("visual_dim", 1024), device=device)
    a = torch.randn(batch_size, sequence_length, config.get("acoustic_dim", 512), device=device)
    t = torch.randn(batch_size, sequence_length, config.get("text_dim", 768), device=device)
    target_latent = torch.randn(
        batch_size, sequence_length, config["hidden_dim"], device=device
    )

    pred_latent, g_music = model(v, a, t)
    g_text = model.encode_text_global(t)

    l_rec = F.mse_loss(pred_latent, target_latent)
    l_align = contrastive_loss(g_music, g_text)
    total_loss = l_rec + config.get("contrastive_loss_weight", 0.5) * l_align

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    print(
        f"Loss: {total_loss.item():.4f} "
        f"(reconstruction: {l_rec.item():.4f}, contrastive: {l_align.item():.4f})"
    )
    print(
        f"Shapes: v={tuple(v.shape)}, a={tuple(a.shape)}, t={tuple(t.shape)}, "
        f"prediction={tuple(pred_latent.shape)}"
    )
    print(f"Device: {device}")
    print("Smoke test passed")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or smoke-test the AMC framework.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_amc.yaml"),
        help="Path to an AMC YAML configuration file.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one synthetic forward/loss/backward/optimizer step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.smoke_test:
        run_smoke_test(config)
        return
    raise SystemExit("Select --smoke-test to run the synthetic AMC validation step.")


if __name__ == "__main__":
    main()
