"""Dual-state Tiny Recursive Transformer for ARC-AGI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.dataset import NUM_COLORS, PAD_VALUE
from src.models.layers import RMSNorm, TransformerBlock3D
from src.models.shape_predictor import GridShapePredictor


@dataclass
class TRMConfig:
    max_grid: int = 30
    max_demos: int = 4
    dim: int = 256
    heads: int = 8
    z_layers: int = 2
    y_layers: int = 1
    outer_steps: int = 6
    inner_steps: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_tasks: int = 512
    max_params: int = 50_000_000


class DualStateRecursiveTransformer(nn.Module):
    """Weight-tied z/y recursive transformer with deep-supervision logits."""

    def __init__(self, config: TRMConfig | None = None):
        super().__init__()
        self.config = config or TRMConfig()
        cfg = self.config
        self.n_cells = cfg.max_grid * cfg.max_grid
        max_pair = cfg.max_demos + 2

        self.color_embed = nn.Embedding(NUM_COLORS + 1, cfg.dim, padding_idx=0)
        self.io_embed = nn.Embedding(2, cfg.dim)
        self.pair_embed = nn.Embedding(max_pair, cfg.dim)
        self.task_embed = nn.Embedding(cfg.num_tasks + 1, cfg.dim)
        self.answer_seed = nn.Parameter(torch.zeros(1, self.n_cells, cfg.dim))
        self.scratch_seed = nn.Parameter(torch.zeros(1, self.n_cells, cfg.dim))

        self.context_blocks = nn.ModuleList(
            [TransformerBlock3D(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout, cfg.max_grid, max_pair) for _ in range(1)]
        )
        self.z_blocks = nn.ModuleList(
            [TransformerBlock3D(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout, cfg.max_grid, max_pair) for _ in range(cfg.z_layers)]
        )
        self.y_blocks = nn.ModuleList(
            [TransformerBlock3D(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout, cfg.max_grid, max_pair) for _ in range(cfg.y_layers)]
        )
        self.fuse_z = nn.Linear(cfg.dim * 3, cfg.dim, bias=False)
        self.fuse_y = nn.Linear(cfg.dim * 2, cfg.dim, bias=False)
        self.norm = RMSNorm(cfg.dim)
        self.output_head = nn.Linear(cfg.dim, NUM_COLORS)
        self.shape_head = GridShapePredictor(cfg.dim, cfg.max_grid)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.normal_(param, std=0.02)
        nn.init.zeros_(self.answer_seed)
        nn.init.zeros_(self.scratch_seed)

    @staticmethod
    def _colors_to_embedding_ids(grid: torch.Tensor) -> torch.Tensor:
        return grid.clamp_min(PAD_VALUE).add(1).clamp_min(0)

    def _grid_positions(self, batch: int, pair_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_grid = self.config.max_grid
        rows = torch.arange(max_grid, device=device).repeat_interleave(max_grid)
        cols = torch.arange(max_grid, device=device).repeat(max_grid)
        pairs = torch.full_like(rows, pair_idx)
        return rows.expand(batch, -1), cols.expand(batch, -1), pairs.expand(batch, -1)

    def _embed_grid(self, grid: torch.Tensor, io_id: int, pair_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = grid.shape[0]
        device = grid.device
        flat = grid.flatten(1)
        emb = self.color_embed(self._colors_to_embedding_ids(flat))
        emb = emb + self.io_embed.weight[io_id].view(1, 1, -1)
        emb = emb + self.pair_embed.weight[pair_idx].view(1, 1, -1)
        row, col, pair = self._grid_positions(bsz, pair_idx, device)
        return emb, row, col, pair

    def _build_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks, rows, cols, pairs, masks = [], [], [], [], []
        max_demos = self.config.max_demos
        for demo_idx in range(max_demos):
            for key, io_id in (("demo_inputs", 0), ("demo_outputs", 1)):
                emb, row, col, pair = self._embed_grid(batch[key][:, demo_idx], io_id, demo_idx)
                chunks.append(emb)
                rows.append(row)
                cols.append(col)
                pairs.append(pair)
                mask_key = "demo_input_masks" if key == "demo_inputs" else "demo_output_masks"
                masks.append(batch[mask_key][:, demo_idx].flatten(1))
        test_emb, row, col, pair = self._embed_grid(batch["test_input"], 0, max_demos)
        chunks.append(test_emb)
        rows.append(row)
        cols.append(col)
        pairs.append(pair)
        masks.append(batch["test_input_mask"].flatten(1))

        x = torch.cat(chunks, dim=1)
        row_pos = torch.cat(rows, dim=1)
        col_pos = torch.cat(cols, dim=1)
        pair_pos = torch.cat(pairs, dim=1)
        valid_mask = torch.cat(masks, dim=1)
        if "task_id" in batch:
            x = x + self.task_embed(batch["task_id"].clamp_max(self.config.num_tasks)).unsqueeze(1)
        for block in self.context_blocks:
            x = block(x, row_pos, col_pos, pair_pos, key_padding_mask=valid_mask)
        return x, row_pos, col_pos, pair_pos, valid_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        cfg = self.config
        context, ctx_row, ctx_col, ctx_pair, ctx_valid = self._build_context(batch)
        bsz = context.shape[0]
        device = context.device
        y = self.answer_seed.expand(bsz, -1, -1)
        z = self.scratch_seed.expand(bsz, -1, -1)
        out_row, out_col, out_pair = self._grid_positions(bsz, cfg.max_demos + 1, device)
        out_valid = torch.ones((bsz, self.n_cells), dtype=torch.bool, device=device)
        logits_by_step = []

        for _ in range(cfg.outer_steps):
            for _inner in range(cfg.inner_steps):
                z_seq = self.fuse_z(torch.cat([context[:, -self.n_cells :], y, z], dim=-1))
                z_full = torch.cat([context, z_seq], dim=1)
                row = torch.cat([ctx_row, out_row], dim=1)
                col = torch.cat([ctx_col, out_col], dim=1)
                pair = torch.cat([ctx_pair, out_pair], dim=1)
                valid = torch.cat([ctx_valid, out_valid], dim=1)
                for block in self.z_blocks:
                    z_full = block(z_full, row, col, pair, key_padding_mask=valid)
                z = z_full[:, -self.n_cells :]

            y_seq = self.fuse_y(torch.cat([y, z], dim=-1))
            for block in self.y_blocks:
                y_seq = block(y_seq, out_row, out_col, out_pair, key_padding_mask=out_valid)
            y = y_seq
            logits_by_step.append(self.output_head(self.norm(y)).view(bsz, cfg.max_grid, cfg.max_grid, NUM_COLORS).permute(0, 3, 1, 2))

        pooled = z.mean(dim=1)
        height_logits, width_logits = self.shape_head(pooled)
        return {
            "logits": logits_by_step[-1],
            "logits_by_step": logits_by_step,
            "height_logits": height_logits,
            "width_logits": width_logits,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
