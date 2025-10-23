"""Implementation of a Wasserstein (WGAN-style) loss for trajectory GFlowNets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gfn.containers import Trajectories
from gfn.env import Env
from gfn.estimators import Estimator
from gfn.gflownet.base import TrajectoryBasedGFlowNet


@dataclass
class WassersteinLossConfig:
    """Configuration container for Wasserstein loss hyper-parameters."""

    gradient_penalty_coef: float = 10.0


class WassersteinGFlowNet(TrajectoryBasedGFlowNet):
    """Trajectory-based GFlowNet trained with a WGAN-style objective."""

    def __init__(
        self,
        pf: Estimator,
        pb: Estimator | None,
        critic: Estimator,
        config: WassersteinLossConfig | None = None,
        constant_pb: bool = False,
    ) -> None:
        super().__init__(pf=pf, pb=pb, constant_pb=constant_pb)
        self.critic = critic
        self.config = config or WassersteinLossConfig()

    # ---------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------
    def critic_parameters(self) -> list[torch.nn.Parameter]:
        """Returns the critic parameters (for optimizer construction)."""

        return list(self.critic.parameters())

    def _trajectory_log_pf(
        self, trajectories: Trajectories, recalculate_all_logprobs: bool
    ) -> torch.Tensor:
        log_pf, _ = self.get_pfs_and_pbs(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        if log_pf is None:
            raise RuntimeError("Forward policy log-probabilities are required")

        lengths = trajectories.terminating_idx
        max_len, batch = log_pf.shape
        mask = (
            torch.arange(max_len, device=log_pf.device).unsqueeze(1)
            < lengths.unsqueeze(0)
        )
        masked_log_pf = torch.where(mask, log_pf, torch.zeros_like(log_pf))
        return masked_log_pf.sum(dim=0)

    def _importance_weights(
        self, log_rewards: torch.Tensor, total_log_pf: torch.Tensor
    ) -> torch.Tensor:
        log_weights = log_rewards - total_log_pf
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        return torch.exp(log_weights)

    def _critic_values(
        self, trajectories: Trajectories, env: Env
    ) -> torch.Tensor:
        states = trajectories.terminating_states
        values = self.critic(states).squeeze(-1)
        return values

    # ------------------------------------------------------------------
    # Gradient penalty helper
    # ------------------------------------------------------------------
    def _gradient_penalty(
        self,
        trajectories: Trajectories,
        importance_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.gradient_penalty_coef <= 0:
            return torch.zeros(1, device=trajectories.device)

        states = trajectories.terminating_states
        embeddings = self.critic.preprocessor(states)
        embeddings = embeddings.view(embeddings.shape[0], -1)

        batch_size = embeddings.shape[0]
        if batch_size == 0:
            return torch.zeros(1, device=embeddings.device)

        with torch.no_grad():
            real_indices = torch.multinomial(
                importance_weights, num_samples=batch_size, replacement=True
            )
        real = embeddings[real_indices]
        fake = embeddings

        epsilon = torch.rand(
            batch_size, 1, device=embeddings.device, dtype=embeddings.dtype
        )
        interpolates = epsilon * real + (1.0 - epsilon) * fake
        interpolates.requires_grad_(True)

        critic_out = self.critic.module(interpolates)
        if critic_out.shape[-1] != 1:
            critic_out = self.critic.reduction_function(critic_out, -1)

        gradients = torch.autograd.grad(
            outputs=critic_out.sum(),
            inputs=interpolates,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        penalty = (gradients.norm(2, dim=1) - 1.0) ** 2
        return penalty.mean()

    # ------------------------------------------------------------------
    # Loss surfaces
    # ------------------------------------------------------------------
    def critic_loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if trajectories.n_trajectories == 0:
            return torch.zeros(1, device=env.device)

        log_rewards = trajectories.log_rewards
        if log_rewards is None:
            raise RuntimeError("Trajectories must contain log rewards for critic loss")

        with torch.no_grad():
            total_log_pf = self._trajectory_log_pf(
                trajectories, recalculate_all_logprobs
            )
            importance_weights = self._importance_weights(log_rewards, total_log_pf)

        critic_outputs = self._critic_values(trajectories, env)
        critic_real = torch.dot(importance_weights, critic_outputs)
        critic_fake = critic_outputs.mean()

        gp = self._gradient_penalty(trajectories, importance_weights.detach())

        return -(critic_real - critic_fake) + self.config.gradient_penalty_coef * gp

    def generator_loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if trajectories.n_trajectories == 0:
            return torch.zeros(1, device=env.device)

        log_rewards = trajectories.log_rewards
        if log_rewards is None:
            raise RuntimeError(
                "Trajectories must contain log rewards for Wasserstein generator loss"
            )

        total_log_pf = self._trajectory_log_pf(
            trajectories, recalculate_all_logprobs
        )
        critic_outputs = self._critic_values(trajectories, env)

        with torch.no_grad():
            importance_weights = self._importance_weights(log_rewards, total_log_pf)
            target_value = torch.dot(importance_weights, critic_outputs)

        advantages = critic_outputs.detach() - target_value
        loss = -(advantages * total_log_pf).mean()
        return loss

    # Keep signature compatibility with training loop.
    def loss(
        self,
        env: Env,
        training_objects: Trajectories,
        recalculate_all_logprobs: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.generator_loss(
            env, training_objects, recalculate_all_logprobs, **kwargs
        )

