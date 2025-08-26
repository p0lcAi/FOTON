# run_from_config.py
from __future__ import annotations

import os
import argparse
import yaml
import torch
import torch.nn as nn

from data_setup import get_data_visually
from model_utils.foton import FOTON
from training_utils import train_test_FOTON
from wandb_helpers import maybe_init_wandb, maybe_log, maybe_finish  # optional but harmless


# ---------- helpers ----------
def get_activation(name: str | None) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    key = (name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
        "leakyrelu": nn.LeakyReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "lrelu": nn.LeakyReLU(),
        "silu": nn.SiLU(),  # swish
        "swish": nn.SiLU(),
        "elu": nn.ELU(),
        "identity": nn.Identity(),
    }.get(key, nn.ReLU())


def get_loss(name: str) -> nn.Module:
    key = name.lower()
    if key in ("cross_entropy", "crossentropy", "ce"):
        return nn.CrossEntropyLoss()
    if key in ("mse", "meansquarederror"):
        return nn.MSELoss()
    raise ValueError(f"Unknown loss '{name}'")


def lr_sched_list(cfg: dict) -> list:
    sch = cfg["training"]["lr_sched"]
    return [sch["factor"], sch["milestones"]]


def thresholds_tuple(cfg: dict) -> tuple[int, float, float]:
    th = cfg["training"]["thresholds"]
    return (th["epoch"], th["test_acc_min"], th["train_acc_min"])


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def build_wandb_config(cfg: dict) -> dict:
    """Map your nested config to the keys wandb_helpers expects."""
    exp = cfg["experiment"]
    return {
        "EXP_NAME": exp["name"],
        "WANDB_PROJECT": exp["project"],
        # "WANDB_ENTITY": "your-team",  # or leave to env WANDB_ENTITY
        "WANDB_ENABLE": os.getenv("WANDB_ENABLE", "true"),
        "WANDB_MODE": os.getenv("WANDB_MODE", "online"),
        "WANDB_TAGS": exp.get("tags", []),
        # also record the full config for reference
        "APP_CONFIG": cfg,
    }


# ---------- main ----------
def main(cfg_path: str):
    # load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # seeds / device
    torch.manual_seed(cfg["experiment"]["seed"])
    device = torch.device(cfg["experiment"]["device"])

    # data
    train_loader, test_loader, _, _, train_data, test_data, input_dim, output_dim = get_data_visually(
        dataset_class=cfg["data"]["dataset"],
        batch_size=cfg["data"]["batch_size"],
        colored=True,
        visually=False,
    )
    
    # model selection 
    family = cfg["model"].get("family", "").lower()
    if family == "conv":
        # infer (C,H,W) from a batch
        sample_x, _ = next(iter(train_loader))
        input_shape = (sample_x.shape[1], sample_x.shape[2], sample_x.shape[3])  # (C,H,W)
        num_classes = output_dim


        arch = cfg["model"]["arch"].lower()
        variant = cfg["model"]["conv"]["variant"]  # "1L" or "2L"

        if arch == "bcop":
            model = FOTONBCOP.from_config(cfg, input_shape=input_shape, num_classes=num_classes).to(device)

        elif arch == "conv":
            model = FOTONConv.from_config(cfg, input_shape=input_shape, num_classes=num_classes).to(device)

        elif arch == "convsk":
            model = FOTONConvSK.from_config(cfg, input_shape=input_shape, num_classes=num_classes).to(device)

        elif arch == "lenet5":
            model = FOTONLeNet5.from_config(cfg, input_shape=input_shape, num_classes=num_classes).to(device)

        else:
            raise ValueError(f"Unknown conv arch: {cfg['model']['arch']}")

    else:
        #  MLP / vanilla FOTON 
        dims = cfg["model"]["layers"]["dims"]
        act = get_activation(cfg["model"]["layers"]["activation"])
        model = FOTON(
            layers_dim=[input_dim] + dims + [output_dim],
            bias=cfg["model"]["layers"]["bias"],
            vision=cfg["data"]["vision"],
            activation_f=act,
            layers_init=cfg["model"]["layers"]["init"],
            update_F=cfg["training"]["update_F"],
            ortho_rate=cfg["model"]["ortho"]["rate"],
            p_dropout=cfg["model"]["layers"]["dropout"],
            error=cfg["model"]["error"],
            ce_T=cfg["training"]["ce_T"],
            L2=cfg["training"]["weight_decay"],
            batch_size=cfg["data"]["batch_size"],
            optimizer=cfg["optim"]["name"].upper(),  # only used for BP reference grads if enabled
        ).to(device)

    # training args
    loss_fn = get_loss(cfg["training"]["loss"])
    lr = cfg["training"]["lr"]
    lr_sched = lr_sched_list(cfg)
    epochs = cfg["training"]["epochs"]
    threshold = thresholds_tuple(cfg)

    # optional W&B (safe no-op if disabled/missing)
    wb_cfg = build_wandb_config(cfg)
    maybe_init_wandb(wb_cfg)

    # train
    results = train_test_FOTON(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        loss_fn=loss_fn,
        optimizer=None,                 # optional; FOTON uses it only for BP reference grads
        lr=lr,
        lr_sched=lr_sched,
        epochs=epochs,
        device=device,
        config=wb_cfg,                  # passed to wandb_helpers
        sweep=False,
        threshold=threshold,
        use_bp_reference=True,          # flip if you don’t need BP comparison
        save_best_to="checkpoints/foton_best.pth",
        restore_best=True,
    )

    # final eval
    test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_acc:.2f}%")
    
    maybe_log({"final_test_acc_pct": test_acc})
    maybe_finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
