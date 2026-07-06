from __future__ import annotations

import copy

import torch
import torch.nn as nn

from hyperlogic_pure.model import PureDRNetModel, continuous_to_bipolar


class CFTeacherDRModel(nn.Module):
    """TabResNet + tête CF + tête règles (base Dry Bean / HyperLogic)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_rules: int = 64,
        cf_hidden_dim: int = 128,
        hyper_hidden_dim: int = 128,
        temperature: float = 0.7,
        tabresnet_n_blocks: int = 4,
        tabresnet_dropout: float = 0.1,
        max_instance_dim: int = 512,
        ctx_hidden_dim: int = 256,
        ctx_modulation: float = 0.1,
        embed_dim_high: int = 128,
    ) -> None:
        super().__init__()
        self.core = PureDRNetModel(
            input_dim,
            num_classes,
            num_rules=num_rules,
            cf_hidden_dim=cf_hidden_dim,
            hyper_hidden_dim=hyper_hidden_dim,
            temperature=temperature,
            tabresnet_n_blocks=tabresnet_n_blocks,
            tabresnet_dropout=tabresnet_dropout,
            max_instance_dim=max_instance_dim,
            ctx_hidden_dim=ctx_hidden_dim,
            ctx_modulation=ctx_modulation,
            embed_dim_high=embed_dim_high,
        )
        self.num_classes = num_classes
        self.num_rules = num_rules
        if self.core.mode == "embed":
            self.core.core.init_cf_modules()

    @property
    def mode(self) -> str:
        return self.core.mode

    @property
    def input_dim_bin(self) -> int:
        return self.core.input_dim_bin

    def _to_model_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "instance":
            return continuous_to_bipolar(x)
        return x.float()

    def forward_rules(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.predict_logits(x)

    def cf_predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Pour chaque classe c : CF vers c, score = logit[c] de la classe c."""
        bsz = x.shape[0]
        nc = self.num_classes
        device = x.device
        x_rep = x.unsqueeze(1).expand(-1, nc, -1).reshape(bsz * nc, x.shape[1])
        y_tgt = torch.arange(nc, device=device).repeat(bsz)
        _, logits_cf = self.core.generate_counterfactual(x_rep, y_tgt)
        logits_cf = logits_cf.view(bsz, nc, nc)
        idx = torch.arange(nc, device=device)
        return logits_cf[:, idx, idx]

    def teacher_cf_logits(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.cf_predict_logits(x)

    def generate_counterfactuals_all_classes(
        self, x: torch.Tensor, y_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.core.generate_counterfactuals_all_classes(x, y_current)

    def set_phase(self, phase: str) -> None:
        """phase in {'cf', 'rules'}."""
        for p in self.parameters():
            p.requires_grad = True

        if self.mode != "instance":
            if phase == "rules" and self.core.core.theta_cf_bias is not None:
                self.core.core.theta_cf_bias.requires_grad = False
            return

        hyper = self.core.core.hyper
        if phase == "cf":
            for p in hyper.cf_head.parameters():
                p.requires_grad = True
            for p in hyper.tab.parameters():
                p.requires_grad = True
            for p in hyper.main_head.parameters():
                p.requires_grad = True
        elif phase == "rules":
            for p in hyper.cf_head.parameters():
                p.requires_grad = False
            for p in hyper.tab.parameters():
                p.requires_grad = not getattr(self, "_freeze_tab_p2", False)
            for p in hyper.main_head.parameters():
                p.requires_grad = True

    def clone_teacher(self) -> CFTeacherDRModel:
        teacher = copy.deepcopy(self)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher
