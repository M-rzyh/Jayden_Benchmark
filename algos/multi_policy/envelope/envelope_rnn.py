"""Envelope Q-Learning implementation."""
import os
from typing import List, Optional, Union
from typing_extensions import override
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
)
from mo_utils.morl_algorithm import MOAgent, RecurrentMOPolicy
from mo_utils.networks import (
    NatureCNN,
    get_grad_norm,
    layer_init,
    mlp,
    polyak_update,
)
from mo_utils.prioritized_buffer import RecurrentPrioritizedReplayBuffer
from mo_utils.utils import linearly_decaying_value, mean_of_unmasked_elements, get_mask_from_dones
from mo_utils.weights import equally_spaced_weights, random_weights
from morl_generalization.generalization_evaluator import MORLGeneralizationEvaluator


class FeaturesExtractor(nn.Module):
    def __init__(self, obs_shape, hidden_dim, rnn_layers=1, recurrent_type='lstm'):
        super().__init__()
        self.obs_shape = obs_shape

        if len(obs_shape) == 1:
            self.state_features = mlp(obs_shape[0], -1, [hidden_dim])
            self.shortcut_state_features = mlp(obs_shape[0], -1, [hidden_dim])
        elif len(obs_shape) > 1:  # Image observation
            self.state_features = NatureCNN(self.obs_shape, features_dim=hidden_dim)
            self.shortcut_state_features = NatureCNN(self.obs_shape, features_dim=hidden_dim)

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

    def _enforce_sequence_dim(self, obs):
        # If obs is not batched (1D), add batch and sequence dimensions
        if len(obs.shape) == len(self.obs_shape):
            obs = obs.unsqueeze(0).unsqueeze(1)  # (1, 1, obs_dim)
        # If obs is batched but not sequenced (2D), add sequence dimension
        elif len(obs.shape) == len(self.obs_shape) + 1:
            obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)
        
        return obs
    
    def _get_shortcut_obs_embedding(self, obs):
        obs = self._enforce_sequence_dim(obs)

        return self.shortcut_state_features(obs)
    
    def _get_rnn_hidden_states(self, obs, hidden=None):
        self.rnn.flatten_parameters()

        obs = self._enforce_sequence_dim(obs)
        sf = self.state_features(obs)
        belief, hidden = self.rnn(sf, hidden)

        return belief, hidden

    def forward(self, obs, hidden=None, return_hidden=True):
        # 1. get hidden/belief states of the whole/sub trajectories, aligned with states
        # return the hidden states (B, T+1, dim)
        belief, hidden = self._get_rnn_hidden_states(obs, hidden)

        # 2. another branch for get shortcut embedding on current obs
        curr_embed = self._get_shortcut_obs_embedding(obs)  # (B, T+1, dim)

        # 3. joint embed
        assert curr_embed.shape == belief.shape
        joint_embeds = th.cat((belief, curr_embed), dim=-1)  # (B, T+1, dim)

        if return_hidden:
            return joint_embeds, hidden
        else:
            return joint_embeds
 
class QNet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, action_dim, rew_dim, input_dim, net_arch):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            input_dim: input dimension (rnn embed dim + shortcut state embed dim)
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        # |S| + |R| -> ... -> |A| * |R|
        self.net = mlp(input_dim + rew_dim, action_dim * rew_dim, net_arch[0:])
        self.apply(layer_init)

    def _enforce_sequence_dim(self, obs, w):
        # If w is not batched (1D), add batch and sequence dimensions
        if w.dim() == 1:
            w = w.unsqueeze(0).unsqueeze(1)  # (1, 1, w_dim)
        # If w is batched but not sequenced (2D), add sequence dimension
        elif w.dim() == 2:
            w = w.unsqueeze(1)  # (batch_size, 1, w_dim)

        # If obs is not batched (1D), add batch and sequence dimensions
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(1)  # (1, 1, obs_dim)
        # If obs is batched but not sequenced (2D), add sequence dimension
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch_size, 1, obs_dim)

        return obs, w

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: latent observation (rnn embed + shortcut state embed)
            w: weight vector

        Returns: the Q values for all actions

        """
        obs, w = self._enforce_sequence_dim(obs, w)

        input = th.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.rew_dim)  # Batch size X Actions X Rewards

def set_requires_grad_flag(net: nn.Module, requires_grad: bool) -> None:
    for p in net.parameters():
        p.requires_grad = requires_grad

def create_target(net: nn.Module) -> nn.Module:
    target = deepcopy(net)
    set_requires_grad_flag(target, False)
    return target

class EnvelopeRNN(RecurrentMOPolicy, MOAgent):
    """Envelope Q-Leaning Algorithm.

    Envelope uses a conditioned network to embed multiple policies (taking the weight as input).
    The main change of this algorithm compare to a scalarized CN DQN is the target update.
    Paper: R. Yang, X. Sun, and K. Narasimhan, “A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation,” arXiv:1908.08342 [cs], Nov. 2019, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1908.08342.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 5,  # ignored if tau != 1.0
        buffer_size: int = 10000,
        net_arch: List = [256, 256],
        rnn_hidden_dim: int = 128,
        rnn_layers: int = 1,
        batch_size: int = 32,
        learning_starts: int = 100,
        gradient_updates: float = 0.1,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = 1.0,
        dist: str = "gaussian",
        num_sample_w: int = 4,
        per: bool = True,
        per_alpha: float = 0.6,
        initial_homotopy_lambda: float = 0.0,
        final_homotopy_lambda: float = 1.0,
        homotopy_decay_steps: int = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "EnvelopeRNN",
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """Envelope Q-learning algorithm adapted for recurrent policies.

        Key differences with recurrent version:
        - Environment should have a `max_episode_steps` during registration as it will be used as a fixed sequence length.
        - Replay buffer size and batch size is in episodes rather than steps. Each episode is of length `max_episode_steps`.
        - Policy updates happen on episodic basis rather than step basis. As such, `target_net_update_freq` should be lower.
        - `gradient_updates` parameter is measured in proportion of steps taken in an episode to update the networks.
        - `self.zero_start_rnn_hidden()` should be called at the beginning of each episode BOTH during training and evaluation
          to zero-start the RNN's hidden state.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate (alpha).
            initial_epsilon: The initial epsilon value for epsilon-greedy exploration.
            final_epsilon: The final epsilon value for epsilon-greedy exploration.
            epsilon_decay_steps: The number of steps to decay epsilon over.
            tau: The soft update coefficient (keep in [0, 1]).
            target_net_update_freq: The frequency with which the target network is updated. Note this frequency is measured in episodes * gradient_updates.
            buffer_size: The size of the replay buffer. Note that the buffer size is the number of episodes.
            net_arch: The size of the hidden layers of the value net.
            rnn_hidden_dim: The size of the hidden layers of the RNN.
            rnn_layers: The number of layers in the RNN.
            batch_size: The size of the batch to sample from the replay buffer. Note that the batch size is the number of episodes.
            learning_starts: The number of steps before learning starts i.e. the agent will be random until learning starts.
            gradient_updates: The proportion of steps taken in an episode to update the networks. Must be 0.0 < gradient_updates <= 1.0.
            gamma: The discount factor (gamma).
            max_grad_norm: The maximum norm for the gradient clipping. If None, no gradient clipping is applied.
            dist: The distribution to sample the weight vectors from. Either 'gaussian' or 'dirichlet'.
            num_sample_w: The number of weight vectors to sample for the envelope target.
            per: Whether to use prioritized experience replay.
            per_alpha: The alpha parameter for prioritized experience replay.
            initial_homotopy_lambda: The initial value of the homotopy parameter for homotopy optimization.
            final_homotopy_lambda: The final value of the homotopy parameter.
            homotopy_decay_steps: The number of steps to decay the homotopy parameter over.
            project_name: The name of the project, for wandb logging.
            experiment_name: The name of the experiment, for wandb logging.
            wandb_entity: The entity of the project, for wandb logging.
            wandb_group: The wandb group to use for logging.
            log: Whether to log to wandb.
            seed: The seed for the random number generator.
            device: The device to use for training.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        RecurrentMOPolicy.__init__(self, device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.per = per
        self.per_alpha = per_alpha
        self.initial_homotopy_lambda = initial_homotopy_lambda
        self.final_homotopy_lambda = final_homotopy_lambda
        self.homotopy_decay_steps = homotopy_decay_steps
        self.rnn_layers = rnn_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.dist = dist
        self.gradient_updates = gradient_updates
        assert 0.0 < gradient_updates <= 1.0, "Gradient updates must be in the range (0.0, 1.0]"

        self.feat_net = FeaturesExtractor(self.observation_shape, self.rnn_hidden_dim, rnn_layers=rnn_layers).to(self.device)
        self.q_net = QNet(self.observation_shape, self.action_dim, self.reward_dim, input_dim=self.rnn_hidden_dim*2, net_arch=net_arch).to(self.device)
        
        self.target_feat_net = create_target(self.feat_net)
        self.target_q_net = create_target(self.q_net)

        self.feat_optim = optim.Adam(self.feat_net.parameters(), lr=self.learning_rate)
        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        self.sequence_length = self.env.spec.max_episode_steps
        assert self.learning_starts >= self.sequence_length * self.batch_size, "Not enough episodes to start replay"
        self.num_sample_w = num_sample_w
        self.homotopy_lambda = self.initial_homotopy_lambda
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
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, wandb_group)

    @override
    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "clip_grad_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "use_envelope": True,
            "num_sample_w": self.num_sample_w,
            "net_arch": self.net_arch,
            "per": self.per,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "initial_homotopy_lambda": self.initial_homotopy_lambda,
            "final_homotopy_lambda": self.final_homotopy_lambda,
            "homotopy_decay_steps": self.homotopy_decay_steps,
            "rnn_layers": self.rnn_layers,
            "rnn_hidden_dim": self.rnn_hidden_dim,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def save(self, save_replay_buffer: bool = True, save_dir: str = "weights/", filename: Optional[str] = None):
        """Save the model and the replay buffer if specified.

        Args:
            save_replay_buffer: Whether to save the replay buffer too.
            save_dir: Directory to save the model.
            filename: filename to save the model.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params["feat_net_state_dict"] = self.feat_net.state_dict()
        saved_params["q_net_state_dict"] = self.q_net.state_dict()

        saved_params["feat_net_optimizer_state_dict"] = self.feat_optim.state_dict()
        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path: str, load_replay_buffer: bool = True):
        """Load the model and the replay buffer if specified.

        Args:
            path: Path to the model.
            load_replay_buffer: Whether to load the replay buffer too.
        """
        params = th.load(path)
        self.feat_net.load_state_dict(params["feat_net_state_dict"])
        self.q_net.load_state_dict(params["q_net_state_dict"])
        self.target_feat_net.load_state_dict(params["feat_net_state_dict"])
        self.target_q_net.load_state_dict(params["q_net_state_dict"])
        self.feat_optim.load_state_dict(params["feat_net_optimizer_state_dict"])
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    @override
    def update(self, num_updates: int = 1):
        critic_losses = []
        for _ in range(num_updates):
            if self.per:
                (
                    b_obs_seq,
                    b_actions_seq,
                    b_rewards_seq,
                    b_next_obs_seq,
                    b_dones_seq,
                    b_inds,
                ) = self.__sample_batch_experiences()
            else:
                (
                    b_obs_seq,
                    b_actions_seq,
                    b_rewards_seq,
                    b_next_obs_seq,
                    b_dones_seq,
                ) = self.__sample_batch_experiences()

            assert b_obs_seq.shape == (self.batch_size, self.sequence_length, *self.observation_shape)

            # termination step should not be masked, agent needs to predict the value of the last state
            s_masks = get_mask_from_dones(b_dones_seq)

            batch_size, seq_len, _ = b_obs_seq.size()

            # Sample weights for scalarization
            sampled_w = (
                th.tensor(random_weights(dim=self.reward_dim, n=self.num_sample_w, dist=self.dist, rng=self.np_random))
                .float()
                .to(self.device)
            )  # sample num_sample_w random weights
            w_repeated = sampled_w.repeat(batch_size, 1)

            # Reshape to (batch_size * num_sample_w, seq_len, reward_dim)
            w = w_repeated.view(batch_size * self.num_sample_w, 1, -1).expand(-1, seq_len, -1)
            
            # Repeat sequences for each weight
            b_obs_seq, b_actions_seq, b_rewards_seq, b_next_obs_seq, b_dones_seq, s_masks = (
                b_obs_seq.repeat(self.num_sample_w, 1, 1).view(batch_size * self.num_sample_w, seq_len, *self.observation_shape),
                b_actions_seq.repeat(self.num_sample_w, 1, 1).view(batch_size * self.num_sample_w, seq_len, 1),
                b_rewards_seq.repeat(self.num_sample_w, 1, 1).view(batch_size * self.num_sample_w, seq_len, self.reward_dim),
                b_next_obs_seq.repeat(self.num_sample_w, 1, 1).view(batch_size * self.num_sample_w, seq_len, *self.observation_shape),
                b_dones_seq.repeat(self.num_sample_w, 1, 1).view(batch_size * self.num_sample_w, seq_len, 1),
                s_masks.repeat(self.num_sample_w, 1, 1).view(batch_size * self.num_sample_w, seq_len, 1),
            )
            assert b_obs_seq.shape == (batch_size * self.num_sample_w, seq_len, *self.observation_shape)

            with th.no_grad():
                b_next_feat_seq = self.feat_net(b_next_obs_seq, return_hidden=False)
                b_next_feat_seq_targ = self.target_feat_net(b_next_obs_seq, return_hidden=False)
                target = self.envelope_target(b_next_feat_seq, b_next_feat_seq_targ, w, sampled_w)

                # TODO: Check if this is correct
                assert target.shape == (batch_size * self.num_sample_w, seq_len, self.reward_dim)
                target_q = b_rewards_seq + (1 - b_dones_seq) * self.gamma * target

            feat_seq = self.feat_net(b_obs_seq, return_hidden=False)
            q_values_seq = self.q_net(feat_seq, w).view(-1, seq_len, self.action_dim, self.reward_dim)
            # Gather the Q-values for the actions taken in the sequences
            b_actions = b_actions_seq.unsqueeze(-1).expand(-1, -1, -1, self.reward_dim)
            q_values = q_values_seq.gather(
                2,
                b_actions.long(),
            )
            q_values = q_values.squeeze(2)

            assert q_values.shape == target_q.shape

            critic_loss = (q_values - target_q)**2

            critic_loss = mean_of_unmasked_elements(critic_loss, s_masks)

            assert critic_loss.shape == ()

            if self.homotopy_lambda > 0:
                wQ = th.einsum("bsr,bsr->bs", q_values, w)
                wTQ = th.einsum("bsr,bsr->bs", target_q, w)
                auxiliary_loss = (wQ - wTQ)**2
                auxiliary_loss = mean_of_unmasked_elements(auxiliary_loss, s_masks.squeeze(-1))
                critic_loss = (1 - self.homotopy_lambda) * critic_loss + self.homotopy_lambda * auxiliary_loss

            self.feat_optim.zero_grad()
            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.global_step % 100 == 0:
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(self.q_net.parameters()).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.feat_optim.step()
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                # many repeated values, take the original batch size
                td_err = (q_values[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                per = th.einsum("bsr,bsr->bs", td_err, w[: len(b_inds)]).abs()
                priority_max = per.max(dim=1).values
                priority_mean = per.mean(dim=1)
                priority = 0.9 * priority_max + 0.1 * priority_mean  # R2D2 method
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.num_episodes % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)
            polyak_update(self.feat_net.parameters(), self.target_feat_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        if self.log and self.global_step % 100 == 0:
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                },
            )
            if self.per:
                wandb.log(
                    {
                        "metrics/mean_priority": np.mean(priority),
                        "metrics/max_priority": np.max(priority),
                        "metrics/mean_td_error_w": per.abs().mean().item(),
                    },
                    commit=False,
                )

    @override
    def eval(self, 
        obs: np.ndarray, 
        w: np.ndarray,
        **kwargs
    ) -> int:
        num_envs = kwargs.get('num_envs', 1)  # Default to 1 if not provided
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)

        self.q_net.eval() # Set the network to evaluation mode
        self.feat_net.eval()
        feat, self.hidden = self.feat_net(obs, self.hidden)
        action = self.max_action(feat, w, num_envs)
        self.q_net.train()
        self.feat_net.train()

        return action

    def act(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Epsilon-greedily select an action given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: an integer representing the action to take.
        """
        feat, self.hidden = self.feat_net(obs, self.hidden)
        if self.np_random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.max_action(feat, w)

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor, num_envs: int = 1) -> int:
        """Select the action with the highest Q-value given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: the action with the highest Q-value.
        """
        q_values = self.q_net(obs, w)
        if num_envs > 1:
            # w has shape (num_envs, num_rewards)
            scalarized_q_values = th.einsum("br,bar->ba", w, q_values)
            max_act = th.argmax(scalarized_q_values, dim=1)
            return max_act.detach().cpu().numpy() # action has shape (num_envs,)
        else:
            scalarized_q_values = th.einsum("r,bar->ba", w, q_values)
            max_act = th.argmax(scalarized_q_values, dim=1)
        
        return max_act.detach().item()

    @th.no_grad()
    def envelope_target(self, obs: th.Tensor, obs_targ: th.Tensor, w: th.Tensor, sampled_w: th.Tensor) -> th.Tensor:
        """Computes the envelope target for the given observation sequence and weight.

        Args:
            obs: Batched sequences of latent observations for main q net (batch_size, seq_len, obs_dim).
            obs: Batched sequences of latent observations for target q net (batch_size, seq_len, obs_dim).
            w: Current weight vector (batch_size, reward_dim).
            sampled_w: Set of sampled weight vectors (num_sampled_weights, reward_dim).
            hidden_states: Initial hidden states for the recurrent network.

        Returns: 
            max_next_q: The envelope target Q-values (batch_size, seq_len, reward_dim).
        """
        batch_size, seq_len, _ = obs.size()
        num_sampled_weights = sampled_w.size(0)

        # Repeat the observations for each sampled weight and reshape
        next_obs = obs.repeat_interleave(num_sampled_weights, dim=0).view(-1, obs.size(-1))
        next_obs_targ = obs_targ.repeat_interleave(num_sampled_weights, dim=0).view(-1, obs_targ.size(-1))

        # Repeat the weights for each sample
        W = sampled_w.repeat(batch_size * seq_len, 1)

        # Batch size X Num sampled weights X Seq len X Num actions X Num objectives
        next_q_values = self.q_net(next_obs, W)
        next_q_values = next_q_values.view(batch_size, num_sampled_weights, seq_len, self.action_dim, self.reward_dim)
        
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = th.einsum("bsr,bwsar->bwsa", w, next_q_values)
        
        # Max Q values for each sampled weight
        max_q, ac = th.max(scalarized_next_q_values, dim=3)
        
        # Max weights in the envelope (taking the max over sampled weights)
        pref = th.argmax(max_q, dim=1)

        # MO Q-values evaluated on the target network
        next_q_values_target = self.target_q_net(next_obs_targ, W)
        next_q_values_target = next_q_values_target.view(batch_size, num_sampled_weights, seq_len, self.action_dim, self.reward_dim)

        # Index the Q-values for the max actions
        max_next_q = next_q_values_target.gather(
            3,
            ac.unsqueeze(3).unsqueeze(4).expand(next_q_values.size(0), next_q_values.size(1), next_q_values.size(2), 1, next_q_values.size(4)),
        ).squeeze(3)
        
        # Index the Q-values for the max sampled weights
        max_next_q = max_next_q.gather(1, pref.unsqueeze(1).unsqueeze(3).expand(max_next_q.size(0), 1, max_next_q.size(2), max_next_q.size(3))).squeeze(1)
        
        return max_next_q

    def train(
        self,
        total_timesteps: int,
        eval_env: Union[gym.Env, MORLGeneralizationEvaluator],
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        weight: Optional[np.ndarray] = None,
        total_episodes: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_freq: int = 10000,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        reset_learning_starts: bool = False,
        verbose: bool = False,
        test_generalization: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps: total number of timesteps to train for.
            eval_env: environment to use for evaluation. If None, it is ignored.
            ref_point: reference point for the hypervolume computation.
            known_pareto_front: known pareto front for the hypervolume computation.
            weight: weight vector. If None, it is randomly sampled every episode (as done in the paper).
            total_episodes: total number of episodes to train for. If None, it is ignored.
            reset_num_timesteps: whether to reset the number of timesteps. Useful when training multiple times.
            eval_freq: policy evaluation frequency (in number of steps).
            num_eval_weights_for_front: number of weights to sample for creating the pareto front when evaluating.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            reset_learning_starts: whether to reset the learning starts. Useful when training multiple times.
            verbose: whether to print the episode info.
            test_generalization (bool): Whether to test generalizability of the model.
        """
        if eval_env is not None:
            assert ref_point is not None, "Reference point must be provided for the hypervolume computation."
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist() if ref_point is not None else None,
                    "known_front": known_pareto_front,
                    "weight": weight.tolist() if weight is not None else None,
                    "total_episodes": total_episodes,
                    "reset_num_timesteps": reset_num_timesteps,
                    "eval_freq": eval_freq,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "reset_learning_starts": reset_learning_starts,
                }
            )

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        self.gradient_updates_count = 0
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step

        num_episodes = 0
        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
        obs, _ = self.env.reset()

        w = weight if weight is not None else random_weights(self.reward_dim, 1, dist=self.dist, rng=self.np_random)
        tensor_w = th.tensor(w).float().to(self.device)

        obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
        action_seq = np.zeros((self.sequence_length, 1))
        reward_seq = np.zeros((self.sequence_length, self.reward_dim))
        next_obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
        done_seq = np.zeros((self.sequence_length, 1))
        index = 0
        episode_steps = 0
        for _ in range(1, total_timesteps + 1):
            if total_episodes is not None and num_episodes == total_episodes:
                break
            
            with th.no_grad():
                if self.global_step < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(th.as_tensor(np.array(obs)).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            self.global_step += 1

            obs_seq[index] = obs
            action_seq[index] = action
            reward_seq[index] = vec_reward
            next_obs_seq[index] = next_obs
            done_seq[index] = int(terminated)
            index = (index + 1) % self.sequence_length
            episode_steps += 1

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                saved_training_hidden = deepcopy(self.hidden)

                if test_generalization:
                    eval_env.eval(self, ref_point=ref_point, global_step=self.global_step)
                else:
                    current_front = [
                        self.policy_eval(eval_env, weights=ew, num_episodes=num_eval_episodes_for_front, log=self.log)[3]
                        for ew in eval_weights
                    ]
                    log_all_multi_policy_metrics(
                        current_front=current_front,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )
                
                # IMPORTANT: reset hidden state back to previous training step's hidden state because `self.hidden` gets
                # manipulated during evaluation
                self.hidden = saved_training_hidden

            if terminated or truncated:
                obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                self.replay_buffer.add(obs_seq, action_seq, reward_seq, next_obs_seq, done_seq)
                obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
                action_seq = np.zeros((self.sequence_length, 1))
                reward_seq = np.zeros((self.sequence_length, self.reward_dim))
                next_obs_seq = np.zeros((self.sequence_length, *self.observation_shape))
                done_seq = np.zeros((self.sequence_length, 1))
                done_seq = np.zeros((self.sequence_length, 1))
                index = 0

                if self.global_step >= self.learning_starts:
                    self.update(num_updates=max(1, int(self.gradient_updates * episode_steps)))
                    self.gradient_updates_count += max(1, int(self.gradient_updates * episode_steps))
                    wandb.log({"charts/gradient_updates_count": self.gradient_updates_count, "global_step": self.global_step})
                
                episode_steps = 0
                self.zero_start_rnn_hidden() # IMPORTANT: reset hidden state after each episode

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step, verbose=verbose)

                if weight is None:
                    w = random_weights(self.reward_dim, 1, dist=self.dist, rng=self.np_random)
                    tensor_w = th.tensor(w).float().to(self.device)

            else:
                obs = next_obs

        if self.log:
            self.close_wandb()