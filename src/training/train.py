"""Minimal training CLI."""

from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ARCDatasetConfig, ARCTaskDataset
from src.models.recursive_trm import DualStateRecursiveTransformer, TRMConfig
from src.training.trainer import ARCTrainer, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="steps between checkpoints (0 = only at the end)")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    ds = ARCTaskDataset(args.data_dir, ARCDatasetConfig(**cfg["data"]))
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    model = DualStateRecursiveTransformer(TRMConfig(**cfg["model"]))
    trainer_cfg = TrainConfig(total_steps=args.steps, **{k: v for k, v in cfg["training"].items() if k != "batch_size"})
    trainer = ARCTrainer(model, trainer_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume}")
    print(f"parameters={model.count_parameters():,} samples={len(ds)}")
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = trainer.train_step(batch)
        if step % 50 == 0:
            print(f"step={step} loss={metrics['loss']:.4f} lr={metrics['lr']:.2e}")
        if args.checkpoint_every and step and step % args.checkpoint_every == 0:
            trainer.save_checkpoint(f"step_{step}.pt")
    trainer.save_checkpoint()


if __name__ == "__main__":
    main()
