"""GPI-PD algorithm."""
import os
import random
from itertools import chain
from typing import Dict, List, Optional, Union
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from mo_utils.buffer import ReplayBuffer
from mo_utils.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from mo_utils.morl_algorithm import MOAgent, RecurrentMOPolicy
from mo_utils.networks import (
    NatureCNN,
    get_grad_norm,
    huber,
    layer_init,
    mlp,
    polyak_update,
)
from mo_utils.prioritized_buffer import RecurrentPrioritizedReplayBuffer
from mo_utils.utils import linearly_decaying_value, unique_tol, mean_of_unmasked_elements
from mo_utils.weights import equally_spaced_weights
from morl_generalization.generalization_evaluator import MORLGeneralizationEvaluator
from algos.multi_policy.linear_support.linear_support import LinearSupport


class FeaturesNet(nn.Module):
    def __init__(self, obs_shape, hidden_dim, rnn_layers=2, recurrent_type='lstm'):
        super().__init__()
        self.obs_shape = obs_shape

        if len(obs_shape) == 1:
            self.state_features = mlp(obs_shape[0], -1, [hidden_dim])
        elif len(obs_shape) > 1:  # Image observation
            self.state_features = NatureCNN(self.obs_shape, features_dim=hidden_dim)

        if recurrent_type == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=rnn_layers)
        elif recurrent_type == 'rnn':
            self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True, num_layers=rnn_layers)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=rnn_layers)
        else:
            raise ValueError(f"{recurrent_type} not recognized")
        
        self.apply(layer_init)
        
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, obs, hidden=None, return_hidden=True):
        self.rnn.flatten_parameters()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        sf = self.state_features(obs)
        summary, hidden = self.rnn(sf, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary

class QNet(nn.Module):
    """Conditioned MO Q network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch, drop_rate=0.01, layer_norm=True):
        """Initialize the net.

        Args:
            obs_shape: The observation shape.
            action_dim: The action dimension.
            rew_dim: The reward dimension.
            net_arch: The network architecture.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.phi_dim = rew_dim

        self.weights_features = mlp(rew_dim, -1, net_arch[:1])

        self.net = mlp(
            net_arch[0], action_dim * rew_dim, net_arch[1:], drop_rate=drop_rate, layer_norm=layer_norm
        )  # 128/128 256 256 256

        self.apply(layer_init)

    def forward(self, sf, w):
        """Forward pass."""
        wf = self.weights_features(w)
        q_values = self.net(sf * wf)

        # (Batch size X Actions X Rewards), (Batch size X hidden states)
        return q_values.view(-1, self.action_dim, self.phi_dim) 

def set_requires_grad_flag(net: nn.Module, requires_grad: bool) -> None:
    for p in net.parameters():
        p.requires_grad = requires_grad

def create_target(net: nn.Module) -> nn.Module:
    target = deepcopy(net)
    set_requires_grad_flag(target, False)
    return target

class GPILSRNN(RecurrentMOPolicy, MOAgent):
    """GPI-LS algorithm adapted with recurrent networks and recurrent experience replay.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 1000,  # ignored if tau != 1.0
        buffer_size: int = 4000,
        net_arch: List = [256, 256, 256, 256],
        num_nets: int = 2,
        rnn_layers: int = 2,
        batch_size: int = 32,
        learning_starts: int = 100,
        gradient_updates: int = 1,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = None,
        use_gpi: bool = True,
        per: bool = True,
        alpha_per: float = 0.6,
        min_priority: float = 0.01,
        drop_rate: float = 0.01,
        layer_norm: bool = True,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "GPI-PD",
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """Initialize the GPI-PD algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate.
            initial_epsilon: The initial epsilon value.
            final_epsilon: The final epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            tau: The soft update coefficient.
            target_net_update_freq: The target network update frequency.
            buffer_size: The size of the replay buffer. Note that the buffer size is the number of episodes.
            net_arch: The network architecture.
            num_nets: The number of networks.
            rnn_layers: The number of RNN layers.
            batch_size: The batch size. Note that the buffer size is the number of episodes.
            learning_starts: The number of steps before learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor.
            max_grad_norm: The maximum gradient norm.
            use_gpi: Whether to use GPI.
            per: Whether to use PER.
            alpha_per: The alpha parameter for PER.
            min_priority: The minimum priority for PER.
            drop_rate: The dropout rate.
            layer_norm: Whether to use layer normalization.
            project_name: The name of the project.
            experiment_name: The name of the experiment.
            wandb_entity: The name of the wandb entity.
            wandb_group: The wandb group to use for logging.
            log: Whether to log.
            seed: The seed for random number generators.
            device: The device to use.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        RecurrentMOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.use_gpi = use_gpi
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.num_nets = num_nets
        self.drop_rate = drop_rate
        self.layer_norm = layer_norm
        self.rnn_layers = rnn_layers

        # Q-Networks
        self.feat_nets = [
            FeaturesNet(
                self.observation_shape,
                net_arch[0],
                rnn_layers=rnn_layers,
            ).to(self.device)
            for _ in range(self.num_nets)
        ]
        self.q_nets = [
            QNet(
                self.observation_shape,
                self.action_dim,
                self.reward_dim,
                net_arch=net_arch,
                drop_rate=drop_rate,
                layer_norm=layer_norm,
            ).to(self.device)
            for _ in range(self.num_nets)
        ]

        self.target_feat_nets = [
            create_target(feature_net) for feature_net in self.feat_nets
        ]
        self.target_q_nets = [
            create_target(q_net) for q_net in self.q_nets
        ]

        self.feat_optim = optim.Adam(chain(*[net.parameters() for net in self.feat_nets]), lr=self.learning_rate)
        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)

        # Prioritized experience replay parameters
        self.per = per
        self.sequence_length = self.env.spec.max_episode_steps
        assert self.learning_starts >= self.sequence_length * self.batch_size, "Not enough episodes to start replay"
        if self.per:
            self.replay_buffer = RecurrentPrioritizedReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
                sequence_length=self.sequence_length,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape, 1, rew_dim=self.reward_dim, max_size=buffer_size, action_dtype=np.uint8
            )
        self.min_priority = min_priority
        self.alpha = alpha_per

        # logging
        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, wandb_group)

    def get_config(self):
        """Return the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps:": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "per": self.per,
            "alpha_per": self.alpha,
            "min_priority": self.min_priority,
            "tau": self.tau,
            "num_nets": self.num_nets,
            "clip_grad_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "drop_rate": self.drop_rate,
            "layer_norm": self.layer_norm,
            "rnn_layers": self.rnn_layers,
            "seed": self.seed,
        }
    
    def save(self, save_replay_buffer=True, save_dir="weights/", filename=None):
        """Save the model parameters and the replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        
        for i, psi_net in enumerate(self.q_nets):
            saved_params[f"psi_net_{i}_state_dict"] = psi_net.state_dict()
        
        for i, feat_net in enumerate(self.feat_nets):
            saved_params[f"feat_net_{i}_state_dict"] = feat_net.state_dict()

        saved_params["feat_nets_optimizer_state_dict"] = self.feat_optim.state_dict()
        saved_params["psi_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        saved_params["M"] = self.weight_support
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the model parameters and the replay buffer."""
        params = th.load(path, map_location=self.device)
        for i, (psi_net, target_psi_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            psi_net.load_state_dict(params[f"psi_net_{i}_state_dict"])
            target_psi_net.load_state_dict(params[f"psi_net_{i}_state_dict"])

        for i, (feat_net, target_feat_net) in enumerate(zip(self.feat_nets, self.target_feat_nets)):
            feat_net.load_state_dict(params[f"feat_net_{i}_state_dict"])
            target_feat_net.load_state_dict(params[f"feat_net_{i}_state_dict"])
            
        self.feat_optim.load_state_dict(params["feat_nets_optimizer_state_dict"])
        self.q_optim.load_state_dict(params["psi_nets_optimizer_state_dict"])
        self.weight_support = params["M"]
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)
    
    def update(self, weight: th.Tensor):
        """Update the parameters of the networks."""
        critic_losses = []
        for _ in range(self.gradient_updates):
            if self.per:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones, idxes = self._sample_batch_experiences()
            else:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = self._sample_batch_experiences()

            num_repeats = 1
            if len(self.weight_support) > 1:
                s_obs, s_actions, s_rewards, s_next_obs, s_dones = (
                    s_obs.repeat(2, *(1 for _ in range(s_obs.dim() - 1))),
                    s_actions.repeat(2, *(1 for _ in range(s_actions.dim() - 1))),
                    s_rewards.repeat(2, *(1 for _ in range(s_rewards.dim() - 1))),
                    s_next_obs.repeat(2, *(1 for _ in range(s_obs.dim() - 1))),
                    s_dones.repeat(2, *(1 for _ in range(s_dones.dim() - 1))),
                )
                # Half of the batch uses the given weight vector, the other half uses weights sampled from the support set
                w = th.vstack(
                    [weight for _ in range(s_obs.size(0) // 2)] + random.choices(self.weight_support, k=s_obs.size(0) // 2)
                )
            else:
                w = weight.repeat(s_obs.size(0), 1)

            w = w.unsqueeze(1).repeat(1, self.sequence_length, 1)
            assert w.shape == (self.batch_size * num_repeats, self.sequence_length, self.reward_dim)

            with th.no_grad():
                # Compute min_i Q_i(s', a, w) . w
                next_features = th.stack([target_feat_net(s_next_obs, return_hidden=False) for target_feat_net in self.target_feat_nets])
                assert next_features.shape == (len(self.target_feat_nets), self.batch_size * num_repeats, self.sequence_length, self.net_arch[0])
                next_q_values = th.stack([
                    target_psi_net(next_features[i], w).view(-1, self.sequence_length, self.action_dim, self.reward_dim)
                    for i, target_psi_net in enumerate(self.target_q_nets)
                ])
                assert next_q_values.shape == (len(self.target_feat_nets), self.batch_size * num_repeats, self.sequence_length, self.action_dim, self.reward_dim)
                scalarized_next_q_values = th.einsum("nbsar,bsr->nbsa", next_q_values, w)  # q_i(s', a, w)
                min_inds = th.argmin(scalarized_next_q_values, dim=0) # b x s x a
                min_inds = min_inds.reshape(1, next_q_values.size(1), next_q_values.size(2), next_q_values.size(3), 1).expand(
                    1, next_q_values.size(1), next_q_values.size(2), next_q_values.size(3), next_q_values.size(4)
                ) # 1 x b x s x a x r
                next_q_values = next_q_values.gather(0, min_inds).squeeze(0) # b x s x a x r

                # Compute max_a Q(s', a, w) . w
                max_q = th.einsum("bsr,bsar->bsa", w, next_q_values)
                max_acts = th.argmax(max_q, dim=2) # b x s

                q_targets = next_q_values.gather(
                    2, max_acts.long().reshape(max_acts.size(0), max_acts.size(1), 1, 1).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3))
                )

                assert q_targets.shape == (self.batch_size * num_repeats, self.sequence_length, self.reward_dim)
                target_q = s_rewards + (1 - s_dones) * self.gamma * target_q

            losses = []
            td_errors = []
            for i, psi_net in enumerate(self.q_nets):
                next_feat = self.feat_nets[i](s_obs, return_hidden=False)
                psi_value = psi_net(next_feat, w).view(-1, self.sequence_length, self.action_dim, self.reward_dim)
                # Gather the Q-values for the actions taken in the sequences
                s_actions_seq = s_actions.unsqueeze(-1).expand(-1, -1, -1, self.reward_dim)
                psi_value = psi_value.gather(
                    2, 
                    s_actions_seq.long()
                )
                psi_value = psi_value.squeeze(2)
                assert psi_value.shape == target_q.shape

                td_error = psi_value - target_q # (b, s, r)
                loss = td_error ** 2
                loss = mean_of_unmasked_elements(loss, (1 - s_dones))
                assert loss.shape == ()
                losses.append(loss)
                if self.per:
                    td_errors.append(td_error.abs())
            critic_loss = (1 / self.num_nets) * sum(losses)

            self.feat_optim.zero_grad()
            self.q_optim.zero_grad()
            critic_loss.backward()

            if self.log and self.global_step % 100 == 0:
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(self.q_nets[0].parameters()).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                for psi_net in self.q_nets:
                    th.nn.utils.clip_grad_norm_(psi_net.parameters(), self.max_grad_norm)
            
            self.feat_optim.step()
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_error = th.max(th.stack(td_errors), dim=0)[0] # (b, s, r)
                td_error = td_error[: len(idxes)].detach()
                per = th.einsum("bsr,bsr->bs", w[: len(idxes)], td_error).abs()
                priority_max = per.max(dim=1).values
                priority_mean = per.mean(dim=1)
                priority = 0.9 * priority_max + 0.1 * priority_mean  # R2D2 method
                priority = priority.cpu().numpy().flatten()
                priority = priority.clip(min=self.min_priority) ** self.alpha

                self.replay_buffer.update_priorities(idxes, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            for psi_net, target_psi_net in zip(self.q_nets, self.target_q_nets):
                polyak_update(psi_net.parameters(), target_psi_net.parameters(), self.tau)
            for feat_net, target_feat_net in zip(self.feat_nets, self.target_feat_nets):
                polyak_update(feat_net.parameters(), target_feat_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon, self.epsilon_decay_steps, self.global_step, self.learning_starts, self.final_epsilon
            )

        if self.log and self.global_step % 100 == 0:
            if self.per:
                wandb.log(
                    {
                        "metrics/mean_priority": np.mean(priority),
                        "metrics/max_priority": np.max(priority),
                        "metrics/mean_td_error_w": per.abs().mean().item(),
                    },
                    commit=False,
                )
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "global_step": self.global_step,
                },
            )

    @th.no_grad()
    def gpi_action(
        self, 
        obs: th.Tensor,
        w: th.Tensor, 
        num_envs: int = 1,
    ) -> Union[int, np.ndarray]:
        """Select an action using GPI."""

        if num_envs > 1:
            M = th.stack(self.weight_support).repeat(num_envs, 1, 1) # (num_envs, num_weights, reward_dim)

            # (num_envs, num_weights, obs_dim1, obs_dim2, ...) Obs can be multi-dimensional images
            obs_m = obs.unsqueeze(1).repeat(1, len(self.weight_support), *(1 for _ in range(obs.dim() - 1)))  
            
            if obs_m.dim() > 4: # image obs would be (num_envs, num_weights, channel, width, height)
                obs_m = obs_m.view(-1, *obs.shape[1:]) # (num_envs * num_weights, channel, width, height)
                M = M.view(-1, M.shape[-1])  # (num_envs * num_weights, reward_dim)
                q_values = self.q_nets[0](obs, M) # (num_envs * num_weights, action_dim, reward_dim)
            else:
                q_values = self.q_nets[0](obs, M) # (num_envs * num_weights, action_dim, reward_dim)

            q_values = q_values.view(num_envs, len(self.weight_support), self.action_dim, -1) # (num_envs, num_weights, action_dim, reward_dim)
            scalar_q_values = th.einsum("br,bpar->bpa", w, q_values) # (num_envs, num_weights, action_dim)
            max_q, a = th.max(scalar_q_values, dim=2) # get best action for each weight, (num_envs, num_weights)
            policy_index = th.argmax(max_q, dim=1) # get best weight with highest scalarized value, (num_envs)
            action = a[th.arange(num_envs), policy_index].detach().cpu().numpy() # choose best action for best weight
        else:
            M = th.stack(self.weight_support)

            obs_m = obs.repeat(M.size(0), *(1 for _ in range(obs.dim())))
            q_values = self.q_nets[0](obs_m, M)
            scalar_q_values = th.einsum("r,bar->ba", w, q_values)  # q(s,a,w_i) = q(s,a,w_i) . w
            max_q, a = th.max(scalar_q_values, dim=1)
            policy_index = th.argmax(max_q)  # max_i max_a q(s,a,w_i)
            action = a[policy_index].detach().item()

        return action

    @th.no_grad()
    def eval(
        self, 
        obs: np.ndarray, 
        w: np.ndarray,
        num_envs: int = 1,
        **kwargs,
    ) -> int:
        """Select an action for the given obs and weight vector."""
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)

        # set to evaluation mode
        for q_net in self.q_nets:
            q_net.eval()
        for feat_net in self.feat_nets:
            feat_net.eval()
        
        feat, self.hidden = self.feat_nets[0](obs, self.hidden)
        if self.use_gpi:
            action = self.gpi_action(feat, w, num_envs=num_envs)
        else:
            action = self.max_action(feat, w, num_envs=num_envs)

        # set back to training mode
        for q_net in self.q_nets:
            q_net.train()
        for feat_net in self.feat_nets:
            feat_net.train()

        return action

    def _act(self, obs: th.Tensor, w: th.Tensor) -> int:
        feat, self.hidden = self.feat_nets[0](obs, self.hidden)
        if self.np_random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if self.use_gpi:
                return self.gpi_action(feat, w)
            else:
                return self.max_action(feat, w)

    @th.no_grad()
    def max_action( # untested for num_envs > 1 currently
        self, 
        obs: th.Tensor, 
        w: th.Tensor,
        num_envs: int = 1,
    ) -> Union[int, np.ndarray]:
        """Select the greedy action."""
        psi = th.min(th.stack([psi_net(obs, w) for psi_net in self.q_nets]), dim=0)[0] # (num_envs, action_dim, reward_dim)
        # psi = self.psi_nets[0](obs, w)
        if num_envs > 1:
            q = th.einsum("br,bar->ba", w, psi) # (num_envs, action_dim)
        else:
            q = th.einsum("r,bar->ba", w, psi) # (1, action_dim)
        max_act = th.argmax(q, dim=1)
        
        if num_envs > 1:
            action = max_act.detach().cpu().numpy()
        else:
            action = max_act.detach().item()

        return action

    @th.no_grad()
    def _reset_priorities(self, w: th.Tensor):
        inds = np.arange(self.replay_buffer.size)
        priorities = np.repeat(0.1, self.replay_buffer.size)
        (
            obs_s,
            actions_s,
            rewards_s,
            next_obs_s,
            dones_s,
        ) = self.replay_buffer.get_all_data(to_tensor=False)
        num_batches = int(np.ceil(obs_s.shape[0] / 1000))
        for i in range(num_batches):
            b = i * 1000
            e = min((i + 1) * 1000, obs_s.shape[0])
            obs, actions, rewards, next_obs, dones = obs_s[b:e], actions_s[b:e], rewards_s[b:e], next_obs_s[b:e], dones_s[b:e]
            obs, actions, rewards, next_obs, dones = (
                th.tensor(obs).to(self.device),
                th.tensor(actions).to(self.device),
                th.tensor(rewards).to(self.device),
                th.tensor(next_obs).to(self.device),
                th.tensor(dones).to(self.device),
            )
            features = self.feat_nets[0](obs, return_hidden=False)
            q_values = self.q_nets[0](features, w.repeat(obs.size(0), obs.size(1), 1))
            actions = actions.unsqueeze(-1).expand(-1, -1, -1, self.reward_dim)
            q_a = q_values.gather(2, actions.long()).squeeze(2)

            next_features = self.feat_nets[0](next_obs, return_hidden=False)
            next_q_values = self.q_nets[0](next_features, w.repeat(next_obs.size(0), next_obs.size(1), 1))
            max_q = th.einsum("r,bsar->bsa", w, next_q_values)
            max_acts = th.argmax(max_q, dim=2)

            target_features = self.target_feat_nets[0](next_obs, return_hidden=False)
            q_targets = self.target_q_nets[0](target_features, w.repeat(next_obs.size(0), next_obs.size(1), 1))
            q_targets = q_targets.gather(
                2, max_acts.long().reshape(max_acts.size(0), max_acts.size(1), 1, 1).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3))
            )
            max_next_q = q_targets.reshape(-1, self.sequence_length, self.reward_dim)

            gtderror = th.einsum("r,bsr->bs", w, (rewards + (1 - dones) * self.gamma * max_next_q - q_a)).abs()
            priority_max = gtderror.max(dim=1).values
            priority_mean = gtderror.mean(dim=1)
            priority = 0.9 * priority_max + 0.1 * priority_mean  # R2D2 method
            priorities[b:e] = priority.clamp(min=self.min_priority).pow(self.alpha).cpu().detach().numpy().flatten()

        self.replay_buffer.update_priorities(inds, priorities)

    def set_weight_support(self, weight_list: List[np.ndarray]):
        """Set the weight support set."""
        weights_no_repeats = unique_tol(weight_list)
        self.weight_support = [th.tensor(w).float().to(self.device) for w in weights_no_repeats]

    def train_iteration(
        self,
        total_timesteps: int,
        weight: np.ndarray,
        weight_support: List[np.ndarray],
        change_w_every_episode: bool = True,
        reset_num_timesteps: bool = True,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 1000,
        reset_learning_starts: bool = False,
        verbose: bool = False
    ):
        """Train the agent for one iteration.

        Args:
            total_timesteps (int): Number of timesteps to train for
            weight (np.ndarray): Weight vector
            weight_support (List[np.ndarray]): Weight support set
            change_w_every_episode (bool): Whether to change the weight vector at the end of each episode
            reset_num_timesteps (bool): Whether to reset the number of timesteps
            eval_env (Optional[gym.Env]): Environment to evaluate on
            eval_freq (int): Number of timesteps between evaluations
            reset_learning_starts (bool): Whether to reset the learning starts
            verbose (bool): whether to print the episode info.
        """
        weight_support = unique_tol(weight_support)  # remove duplicates
        self.set_weight_support(weight_support)
        tensor_w = th.tensor(weight).float().to(self.device)

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        if self.per and len(self.replay_buffer) > 0:
            self._reset_priorities(tensor_w)

        obs, info = self.env.reset()

        obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
        action_seq = np.zeros((self.sequence_length, 1))
        reward_seq = np.zeros((self.sequence_length, self.reward_dim))
        next_obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
        done_seq = np.zeros((self.sequence_length, 1))
        index = 0
        self.reinitialize_hidden() # IMPORTANT: reset hidden state because there can be stale hidden states from recent eval
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            action = self._act(th.as_tensor(obs).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            obs_seq[index] = obs
            action_seq[index] = action
            reward_seq[index] = vec_reward
            next_obs_seq[index] = next_obs
            done_seq[index] = int(terminated)
            index = (index + 1) % self.sequence_length

            if self.global_step >= self.learning_starts:
                self.update(tensor_w)

            if terminated or truncated:
                obs, _ = self.env.reset()
                self.num_episodes += 1
                
                self.replay_buffer.add(obs_seq, action_seq, reward_seq, next_obs_seq, done_seq)
                obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
                action_seq = np.zeros((self.sequence_length, 1))
                reward_seq = np.zeros((self.sequence_length, self.reward_dim))
                next_obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
                done_seq = np.zeros((self.sequence_length, 1))
                index = 0

                self.reinitialize_hidden() # IMPORTANT: reset hidden state after each episode

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, weight, self.global_step, verbose=verbose)

                if change_w_every_episode:
                    weight = random.choice(weight_support)
                    tensor_w = th.tensor(weight).float().to(self.device)
            else:
                obs = next_obs

    def train(
        self,
        total_timesteps: int,
        eval_env: Union[gym.Env, MORLGeneralizationEvaluator],
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        timesteps_per_iter: int = 10000,
        weight_selection_algo: str = "gpi-ls",
        eval_freq: int = 1000,
        eval_mo_freq: int = 10000,
        checkpoints: bool = False,
        verbose: bool = False,
        test_generalization: bool = False,
    ):
        """Train agent.

        Args:
            total_timesteps (int): Number of timesteps to train for.
            eval_env (gym.Env): Environment to evaluate on.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front if known.
            num_eval_weights_for_front: Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            timesteps_per_iter (int): Number of timesteps to train for per iteration.
            weight_selection_algo (str): Weight selection algorithm to use.
            eval_freq (int): Number of timesteps between evaluations.
            eval_mo_freq (int): Number of timesteps between multi-objective evaluations.
            checkpoints (bool): Whether to save checkpoints.
            verbose (bool): whether to print the episode info.
            test_generalization (bool): Whether to test generalizability of the model.
        """
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "timesteps_per_iter": timesteps_per_iter,
                    "weight_selection_algo": weight_selection_algo,
                    "eval_freq": eval_freq,
                    "eval_mo_freq": eval_mo_freq,
                }
            )
        max_iter = total_timesteps // timesteps_per_iter
        linear_support = LinearSupport(num_objectives=self.reward_dim, epsilon=0.0 if weight_selection_algo == "ols" else None)

        weight_history = []

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        for iter in range(1, max_iter + 1):
            if weight_selection_algo == "ols" or weight_selection_algo == "gpi-ls":
                if weight_selection_algo == "gpi-ls":
                    self.set_weight_support(linear_support.get_weight_support())
                    w = linear_support.next_weight(
                        algo="gpi-ls", gpi_agent=self, env=eval_env, rep_eval=num_eval_episodes_for_front
                    )
                else:
                    w = linear_support.next_weight(algo="ols")

                if w is None:
                    break
            else:
                raise ValueError(f"Unknown algorithm {weight_selection_algo}.")

            print("Next weight vector:", w)
            weight_history.append(w)
            if weight_selection_algo == "gpi-ls":
                M = linear_support.get_weight_support() + linear_support.get_corner_weights(top_k=4) + [w]
            elif weight_selection_algo == "ols":
                M = linear_support.get_weight_support() + [w]
            else:
                M = None

            self.train_iteration(
                total_timesteps=timesteps_per_iter,
                weight=w,
                weight_support=M,
                change_w_every_episode=weight_selection_algo == "gpi-ls",
                eval_env=eval_env,
                eval_freq=eval_freq,
                reset_num_timesteps=False,
                reset_learning_starts=False,
                verbose=verbose,
            )

            if weight_selection_algo == "ols":
                value = policy_evaluation_mo(self, eval_env, w, rep=num_eval_episodes_for_front)[3]
                linear_support.add_solution(value, w)
            elif weight_selection_algo == "gpi-ls":
                for wcw in M:
                    n_value = policy_evaluation_mo(self, eval_env, wcw, rep=num_eval_episodes_for_front)[3]
                    linear_support.add_solution(n_value, wcw)

            if self.log and self.global_step % eval_mo_freq == 0:
                # Evaluation
                if test_generalization:
                    eval_env.eval(self, ref_point=ref_point, reward_dim=self.reward_dim, global_step=self.global_step)
                else:
                    gpi_returns_test_tasks = [
                        policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                    ]
                    log_all_multi_policy_metrics(
                        current_front=gpi_returns_test_tasks,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )
                    # This is the EU computed in the paper
                    mean_gpi_returns_test_tasks = np.mean(
                        [np.dot(ew, q) for ew, q in zip(eval_weights, gpi_returns_test_tasks)], axis=0
                    )
                    wandb.log({"eval/Mean Utility - GPI": mean_gpi_returns_test_tasks, "iteration": iter})

            if checkpoints:
                self.save(filename=f"GPI-PD {weight_selection_algo} iter={iter}", save_replay_buffer=False)

        if self.log:
            self.close_wandb()
