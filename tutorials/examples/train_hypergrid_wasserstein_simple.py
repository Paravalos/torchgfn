#!/usr/bin/env python
"""
A minimal HyperGrid training script using the Wasserstein GFlowNet loss.

This mirrors :mod:`train_hypergrid_simple` but swaps the Trajectory Balance objective
for the WGAN-style Wasserstein loss that leverages a scalar critic.

Example usage:
python train_hypergrid_wasserstein_simple.py --ndim 2 --height 8 --epsilon 0.1

Key differences from the TB variant:
- Alternating critic/generator updates following the WGAN recipe
- Scalar critic network with gradient penalty regularisation
- Separate learning rates for critic and generator parameters
"""

from __future__ import annotations

import argparse
from typing import cast

import torch
from tqdm import tqdm

from gfn.estimators import DiscretePolicyEstimator, ScalarEstimator
from gfn.gflownet import WassersteinGFlowNet
from gfn.gflownet.wasserstein import WassersteinLossConfig
from gfn.gym import HyperGrid
from gfn.preprocessors import KHotPreprocessor
from gfn.samplers import Sampler
from gfn.states import DiscreteStates
from gfn.utils.common import set_seed
from gfn.utils.modules import DiscreteUniform, MLP
from gfn.utils.training import validate


def build_estimators(args, env: HyperGrid, preprocessor: KHotPreprocessor):
    """Construct forward, backward, and critic estimators."""

    module_pf = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=env.n_actions,
    )
    if args.uniform_pb:
        module_pb = DiscreteUniform(output_dim=env.n_actions - 1)
    else:
        module_pb = MLP(
            input_dim=preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            trunk=module_pf.trunk,
        )

    pf_estimator = DiscretePolicyEstimator(
        module_pf, env.n_actions, preprocessor=preprocessor, is_backward=False
    )
    pb_estimator = DiscretePolicyEstimator(
        module_pb, env.n_actions, preprocessor=preprocessor, is_backward=True
    )

    critic_module = MLP(
        input_dim=preprocessor.output_dim,
        output_dim=1,
        hidden_dim=args.critic_hidden_dim,
    )
    critic_estimator = ScalarEstimator(
        critic_module,
        preprocessor=preprocessor,
        reduction="mean",
    )

    return pf_estimator, pb_estimator, critic_estimator


def main(args):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    env = HyperGrid(
        ndim=args.ndim,
        height=args.height,
        reward_fn_str="original",
        reward_fn_kwargs={
            "R0": args.R0,
            "R1": args.R1,
            "R2": args.R2,
        },
        device=device,
        calculate_partition=True,
        store_all_states=True,
        check_action_validity=__debug__,
    )
    preprocessor = KHotPreprocessor(height=env.height, ndim=env.ndim)

    pf_estimator, pb_estimator, critic_estimator = build_estimators(
        args, env, preprocessor
    )
    gflownet = WassersteinGFlowNet(
        pf=pf_estimator,
        pb=pb_estimator,
        critic=critic_estimator,
        config=WassersteinLossConfig(
            gradient_penalty_coef=args.gradient_penalty_coef,
        ),
    )

    sampler = Sampler(estimator=pf_estimator)
    gflownet = gflownet.to(device)

    generator_parameters = gflownet.pf_pb_parameters()
    generator_optimizer = torch.optim.Adam(generator_parameters, lr=args.lr)
    critic_optimizer = torch.optim.Adam(
        gflownet.critic_parameters(), lr=args.critic_lr
    )

    visited_terminating_states = env.states_from_batch_shape((0,))
    validation_info = {"l1_dist": float("inf")}
    discovered_modes = set()

    pbar = tqdm(range(args.n_iterations), dynamic_ncols=True)
    for iteration in pbar:
        critic_losses = []
        for _ in range(args.critic_steps):
            critic_trajectories = sampler.sample_trajectories(
                env,
                n=args.batch_size,
                save_logprobs=True,
                save_estimator_outputs=False,
                epsilon=args.epsilon,
            )
            critic_optimizer.zero_grad()
            loss_critic = gflownet.critic_loss(
                env,
                critic_trajectories,
                recalculate_all_logprobs=False,
            )
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                gflownet.critic_parameters(), args.grad_clip
            )
            critic_optimizer.step()
            critic_losses.append(loss_critic.item())

        trajectories = sampler.sample_trajectories(
            env,
            n=args.batch_size,
            save_logprobs=True,
            save_estimator_outputs=False,
            epsilon=args.epsilon,
        )
        visited_terminating_states.extend(
            cast(DiscreteStates, trajectories.terminating_states)
        )

        generator_optimizer.zero_grad()
        loss_generator = gflownet.generator_loss(
            env,
            trajectories,
            recalculate_all_logprobs=False,
        )
        loss_generator.backward()
        gflownet.assert_finite_gradients()
        torch.nn.utils.clip_grad_norm_(generator_parameters, args.grad_clip)
        generator_optimizer.step()
        gflownet.assert_finite_parameters()

        if (iteration + 1) % args.validation_interval == 0:
            validation_info, _ = validate(
                env,
                gflownet,
                args.validation_samples,
                visited_terminating_states,
            )

            assert isinstance(visited_terminating_states, DiscreteStates)
            modes_found = env.modes_found(visited_terminating_states)
            discovered_modes.update(modes_found)

            message = f"Iter {iteration + 1}: "
            if "l1_dist" in validation_info:
                message += f"L1 distance={validation_info['l1_dist']:.8f} "
            message += f"modes discovered={len(discovered_modes)} / {env.n_modes} "
            message += f"n terminating states {len(visited_terminating_states)}"
            print(message)

        avg_critic_loss = sum(critic_losses) / max(len(critic_losses), 1)
        pbar.set_postfix(
            {
                "generator_loss": loss_generator.item(),
                "critic_loss": avg_critic_loss,
                "trajectories_sampled": (iteration + 1)
                * args.batch_size
                * (1 + args.critic_steps),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")
    parser.add_argument("--ndim", type=int, default=2, help="HyperGrid dimensionality")
    parser.add_argument("--height", type=int, default=64, help="HyperGrid height")
    parser.add_argument("--R0", type=float, default=0.1, help="Environment R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment R2")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Generator (pf/pb) learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-4, help="Critic learning rate"
    )
    parser.add_argument(
        "--critic_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the critic network",
    )
    parser.add_argument(
        "--uniform_pb", action="store_true", help="Use a uniform backward policy"
    )
    parser.add_argument(
        "--gradient_penalty_coef",
        type=float,
        default=10.0,
        help="Gradient penalty coefficient for the critic",
    )
    parser.add_argument(
        "--critic_steps",
        type=int,
        default=5,
        help="Number of critic updates per generator step",
    )
    parser.add_argument(
        "--n_iterations", type=int, default=1000, help="Number of training iterations"
    )
    parser.add_argument(
        "--validation_interval", type=int, default=100, help="Validation interval"
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=200000,
        help="Number of validation samples for probability estimation",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration parameter for epsilon-greedy sampling",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold for both generator and critic",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
