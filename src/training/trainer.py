"""Mixed-precision trainer with EMA and checkpointing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW

from src.training.loss import deep_supervision_loss, shape_loss
from src.training.scheduler import cosine_with_warmup


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 200
    total_steps: int = 10_000
    grad_clip: float = 1.0
    shape_loss_weight: float = 0.05
    ema_decay: float = 0.999
    checkpoint_dir: str = "checkpoints"
    amp: bool = True


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def averaged(self, model: torch.nn.Module):
        """Temporarily swap the EMA weights into ``model`` (for evaluation)."""

        with torch.no_grad():
            self._backup = {name: p.detach().clone() for name, p in model.named_parameters() if name in self.shadow}
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name].to(param.device, param.dtype))
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self._backup:
                        param.copy_(self._backup[name])
            self._backup = {}


class ARCTrainer:
    def __init__(self, model: torch.nn.Module, config: TrainConfig | None = None, device: str | torch.device = "cuda"):
        self.model = model
        self.config = config or TrainConfig()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = cosine_with_warmup(self.optimizer, self.config.warmup_steps, self.config.total_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.amp and self.device.type == "cuda")
        self.ema = EMA(model, self.config.ema_decay)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            outputs = self.model(batch)
            loss = deep_supervision_loss(outputs["logits_by_step"], batch["test_output"], batch["test_output_mask"])
            loss = loss + self.config.shape_loss_weight * shape_loss(outputs["height_logits"], outputs["width_logits"], batch["output_shape"])
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema.update(self.model)
        return {"loss": float(loss.detach().cpu()), "lr": self.scheduler.get_last_lr()[0]}

    def save_checkpoint(self, name: str = "latest.pt") -> Path:
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.shadow,
            "config": self.config,
        }
        out = path / name
        torch.save(ckpt, out)
        return out

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> None:
        """Restore model, EMA, and (optionally) optimizer/scheduler state."""

        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            self.ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema"].items()}
        if load_optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
