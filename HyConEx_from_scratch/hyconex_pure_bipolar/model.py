from __future__ import annotations



from dataclasses import dataclass



import torch

import torch.nn as nn



from hyconex_pure_bipolar.bipolar import bipolar_to_continuous, continuous_to_bipolar

from hyconex_pure_local.model import HyConExLocalModel, LocalHypernetPack

from nouveau_module.main_rule_net import main_logits_from_weights





def _main_theta_dim(input_dim: int, num_rules: int, num_classes: int) -> int:

    return input_dim * num_rules + num_rules + num_rules * num_classes + num_classes





@dataclass

class BipolarRulesForwardPack:

    logits: torch.Tensor

    logits_hyper: torch.Tensor

    logits_rules: torch.Tensor

    rule_activations: torch.Tensor

    hyper_pack: LocalHypernetPack

    theta_main: torch.Tensor

    rule_params: tuple[torch.Tensor, ...]

    x_bin: torch.Tensor

    dr_input: torch.Tensor

    rules_on_input: bool





class HyConExBipolarRulesModel(HyConExLocalModel):

    """

    HyConEx pur + DR-Net en entree {-1,+1}.



    - Entree API : x_bin in {-1,+1} (one-hot bipolarise)

    - Si dim <= max_drnet_input_dim : DR-Net directement sur oh_*

    - Sinon : DR-Net sur bipolar(z)

    - CF : generateur HyConEx en [0,1] (comme pure_rules), evalue tel quel

    """



    def __init__(

        self,

        input_dim: int,

        num_classes: int,

        latent_dim: int,

        hidden_dim: int,

        *,

        num_rules: int = 48,

        temperature: float = 0.5,

        hyper_weight: float = 0.78,

        rule_weight: float = 0.22,

        ctx_modulation: float = 0.1,

        max_drnet_input_dim: int = 512,

    ) -> None:

        super().__init__(input_dim, num_classes, latent_dim, hidden_dim)

        self.num_rules = num_rules

        self.temperature = temperature

        self.hyper_weight = hyper_weight

        self.rule_weight = rule_weight

        self.ctx_modulation = ctx_modulation

        self.max_drnet_input_dim = max_drnet_input_dim

        self.rules_on_input = input_dim <= max_drnet_input_dim

        self.dr_input_dim = input_dim if self.rules_on_input else latent_dim



        theta_dim = _main_theta_dim(self.dr_input_dim, num_rules, num_classes)

        self.theta_bias = nn.Parameter(torch.zeros(theta_dim))

        self.ctx_to_theta = nn.Linear(self.dr_input_dim, theta_dim)



    @staticmethod

    def _is_bipolar(x: torch.Tensor) -> bool:

        return bool(x.min().item() < -0.01)



    def _ensure_bipolar(self, x: torch.Tensor) -> torch.Tensor:

        x = x.float()

        if x.min() >= -1.01 and x.max() <= 1.01:

            return torch.where(x > 0.25, torch.ones_like(x), -torch.ones_like(x))

        return torch.where(x > 0.5, torch.ones_like(x), -torch.ones_like(x))



    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        """enc_in in [0,1] pour l'encodeur ; x_bin in {-1,+1} pour DR-Net instance."""

        x = x.float()

        if self._is_bipolar(x):

            x_bin = self._ensure_bipolar(x)

            enc_in = (x_bin + 1.0) * 0.5

        else:

            enc_in = x.clamp(0.0, 1.0)

            x_bin = continuous_to_bipolar(enc_in)

        return enc_in, x_bin



    def _dr_input(self, x_bin: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        if self.rules_on_input:

            return x_bin

        return self._ensure_bipolar(z)



    def _theta_for_batch(self, dr_in: torch.Tensor) -> torch.Tensor:

        if self.training and dr_in.shape[0] <= 256:

            ctx = dr_in.mean(dim=0)

            delta = self.ctx_to_theta(ctx)

            return self.theta_bias + self.ctx_modulation * delta

        return self.theta_bias



    def _hyper_pack_from_enc(self, enc_in: torch.Tensor) -> LocalHypernetPack:

        z = self.encoder(enc_in)

        params = self.hyper(z)

        params = params.view(-1, self.num_classes, self.latent_dim + 1)

        w = params[:, :, : self.latent_dim]

        b = params[:, :, self.latent_dim]

        contributions = w * z.unsqueeze(1)

        logits = contributions.sum(dim=2) + b

        return LocalHypernetPack(z=z, weights=w, bias=b, contributions=contributions, logits=logits)



    def forward_pack(self, x: torch.Tensor) -> BipolarRulesForwardPack:

        enc_in, x_bin = self._split_input(x)

        hyper_pack = self._hyper_pack_from_enc(enc_in)

        dr_in = self._dr_input(x_bin, hyper_pack.z)

        theta = self._theta_for_batch(dr_in)

        logits_rules, rule_act, rule_params = main_logits_from_weights(

            dr_in,

            theta,

            input_dim=self.dr_input_dim,

            num_rules=self.num_rules,

            num_classes=self.num_classes,

            temperature=self.temperature,

        )

        logits = self.hyper_weight * hyper_pack.logits + self.rule_weight * logits_rules

        return BipolarRulesForwardPack(

            logits=logits,

            logits_hyper=hyper_pack.logits,

            logits_rules=logits_rules,

            rule_activations=rule_act,

            hyper_pack=hyper_pack,

            theta_main=theta,

            rule_params=rule_params,

            x_bin=x_bin,

            dr_input=dr_in,

            rules_on_input=self.rules_on_input,

        )



    def local_hypernet_pack(self, x: torch.Tensor) -> LocalHypernetPack:

        x = x.float()

        if x.requires_grad:

            if self._is_bipolar(x):

                enc_in = (x + 1.0) * 0.5

            else:

                enc_in = x.clamp(0.0, 1.0)

        else:

            enc_in, _ = self._split_input(x)

        return self._hyper_pack_from_enc(enc_in)



    def generate_counterfactual(self, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:

        """Generateur HyConEx identique a pure_rules : sortie continue [0,1]."""

        _, x_bin = self._split_input(x)

        x_cont = bipolar_to_continuous(x_bin)

        return super(HyConExLocalModel, self).generate_counterfactual(x_cont, y_target)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.forward_pack(x).logits


