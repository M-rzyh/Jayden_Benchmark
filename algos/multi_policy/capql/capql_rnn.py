"""CAPQL algorithm."""
import os
import random
from itertools import chain
from typing import List, Optional, Union
from collections import namedtuple
from copy import deepcopy

import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal, Independent

from mo_utils.evaluation import (
    log_all_multi_policy_metrics,
    log_episode_info,
    policy_evaluation_mo,
)
from mo_utils.morl_algorithm import MOAgent, RecurrentMOPolicy
from mo_utils.networks import layer_init, mlp, polyak_update
from mo_utils.weights import equally_spaced_weights
from morl_generalization.generalization_evaluator import MORLGeneralizationEvaluator

import gymnasium as gym

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6
RecurrentBatch = namedtuple('RecurrentBatch', 'o a r w d m')


def as_probas(positive_values: np.array) -> np.array:
    return positive_values / np.sum(positive_values)


def as_tensor_on_device(np_array: np.array, device: Optional[Union[th.device, str]] = None) -> th.Tensor:
    return th.tensor(np_array).float().to(device)


class RecurrentReplayBuffer:

    """Use this version when num_bptt == max_episode_len"""

    def __init__(
        self,
        o_dim,
        a_dim,
        r_dim, # reward dimension
        capacity,
        batch_size,
        max_episode_len,  # this will also serve as num_bptt
        segment_len=None,  # for non-overlapping truncated bptt, maybe need a large batch size
        device=None,
    ):

        # placeholders
        self.o = np.zeros((capacity, max_episode_len + 1, o_dim))
        self.a = np.zeros((capacity, max_episode_len, a_dim))
        self.r = np.zeros((capacity, max_episode_len, r_dim))
        self.w = np.zeros((capacity, max_episode_len, r_dim))
        self.d = np.zeros((capacity, max_episode_len, 1))
        self.m = np.zeros((capacity, max_episode_len, 1))
        self.ep_len = np.zeros((capacity,))
        self.ready_for_sampling = np.zeros((capacity,))

        # pointers
        self.episode_ptr = 0
        self.time_ptr = 0

        # trackers
        self.starting_new_episode = True
        self.num_episodes = 0

        # hyper-parameters
        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.r_dim = r_dim
        self.batch_size = batch_size

        self.max_episode_len = max_episode_len

        if segment_len is not None:
            assert max_episode_len % segment_len == 0  # e.g., if max_episode_len = 1000, then segment_len = 100 is ok

        self.segment_len = segment_len
        self.device = device

    def push(self, o, a, r, w, no, d, cutoff): # obs, action, reward, weight, next_obs, done, cutoff

        # zero-out current slot at the beginning of an episode
        if self.starting_new_episode:

            self.o[self.episode_ptr] = 0
            self.a[self.episode_ptr] = 0
            self.r[self.episode_ptr] = 0
            self.w[self.episode_ptr] = 0
            self.d[self.episode_ptr] = 0
            self.m[self.episode_ptr] = 0
            self.ep_len[self.episode_ptr] = 0
            self.ready_for_sampling[self.episode_ptr] = 0

            self.starting_new_episode = False

        # fill placeholders
        self.o[self.episode_ptr, self.time_ptr] = o
        self.a[self.episode_ptr, self.time_ptr] = a
        self.r[self.episode_ptr, self.time_ptr] = r
        self.w[self.episode_ptr, self.time_ptr] = w
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1
        self.ep_len[self.episode_ptr] += 1

        if d or cutoff:

            # fill placeholders
            self.o[self.episode_ptr, self.time_ptr+1] = no
            self.ready_for_sampling[self.episode_ptr] = 1

            # reset pointers
            self.episode_ptr = (self.episode_ptr + 1) % self.capacity
            self.time_ptr = 0

            # update trackers
            self.starting_new_episode = True
            if self.num_episodes < self.capacity:
                self.num_episodes += 1

        else:
            # update pointers
            self.time_ptr += 1

    def sample(self):

        assert self.num_episodes >= self.batch_size

        # sample episode indices

        options = np.where(self.ready_for_sampling == 1)[0]
        ep_lens_of_options = self.ep_len[options]
        probas_of_options = as_probas(ep_lens_of_options)
        choices = np.random.choice(options, p=probas_of_options, size=self.batch_size)

        ep_lens_of_choices = self.ep_len[choices]

        if self.segment_len is None:
            # grab the corresponding numpy array
            # and save computational effort for lstm
            max_ep_len_in_batch = int(np.max(ep_lens_of_choices))

            o = self.o[choices][:, :max_ep_len_in_batch+1, :]
            a = self.a[choices][:, :max_ep_len_in_batch, :]
            r = self.r[choices][:, :max_ep_len_in_batch, :]
            w = self.w[choices][:, :max_ep_len_in_batch, :]
            d = self.d[choices][:, :max_ep_len_in_batch, :]
            m = self.m[choices][:, :max_ep_len_in_batch, :]
            # convert to tensors on the right device
            o = as_tensor_on_device(o, self.device).view(self.batch_size, max_ep_len_in_batch+1, self.o_dim)
            a = as_tensor_on_device(a, self.device).view(self.batch_size, max_ep_len_in_batch, self.a_dim)
            r = as_tensor_on_device(r, self.device).view(self.batch_size, max_ep_len_in_batch, self.r_dim)
            w = as_tensor_on_device(w, self.device).view(self.batch_size, max_ep_len_in_batch, self.r_dim)
            d = as_tensor_on_device(d, self.device).view(self.batch_size, max_ep_len_in_batch, 1)
            m = as_tensor_on_device(m, self.device).view(self.batch_size, max_ep_len_in_batch, 1)

            return RecurrentBatch(o, a, r, w, d, m)

        else:
            num_segments_for_each_item = np.ceil(ep_lens_of_choices / self.segment_len).astype(int)
            o = self.o[choices]
            a = self.a[choices]
            r = self.r[choices]
            w = self.w[choices]
            d = self.d[choices]
            m = self.m[choices]

            o_seg = np.zeros((self.batch_size, self.segment_len + 1, self.o_dim))
            a_seg = np.zeros((self.batch_size, self.segment_len, self.a_dim))
            r_seg = np.zeros((self.batch_size, self.segment_len, self.r_dim))
            w_seg = np.zeros((self.batch_size, self.segment_len, self.r_dim))
            d_seg = np.zeros((self.batch_size, self.segment_len, 1))
            m_seg = np.zeros((self.batch_size, self.segment_len, 1))

            for i in range(self.batch_size):
                start_idx = np.random.randint(num_segments_for_each_item[i]) * self.segment_len
                o_seg[i] = o[i][start_idx:start_idx + self.segment_len + 1]
                a_seg[i] = a[i][start_idx:start_idx + self.segment_len]
                r_seg[i] = r[i][start_idx:start_idx + self.segment_len]
                d_seg[i] = d[i][start_idx:start_idx + self.segment_len]
                m_seg[i] = m[i][start_idx:start_idx + self.segment_len]

            o_seg = as_tensor_on_device(o_seg, self.device)
            a_seg = as_tensor_on_device(a_seg, self.device)
            r_seg = as_tensor_on_device(r_seg, self.device)
            w_seg = as_tensor_on_device(r_seg, self.device)
            d_seg = as_tensor_on_device(d_seg, self.device)
            m_seg = as_tensor_on_device(m_seg, self.device)

            return RecurrentBatch(o_seg, a_seg, r_seg, w_seg, d_seg, m_seg)


class ReplayMemory:
    """Replay memory."""

    def __init__(self, capacity: int):
        """Initialize the replay memory."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, weights, reward, next_state, done):
        """Push a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            np.array(state).copy(),
            np.array(action).copy(),
            np.array(weights).copy(),
            np.array(reward).copy(),
            np.array(next_state).copy(),
            np.array(done).copy(),
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, to_tensor=True, device=None):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, w, reward, next_state, done = map(np.stack, zip(*batch))
        experience_tuples = (state, action, w, reward, next_state, done)
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x, dtype=th.float32).to(device), experience_tuples))
        return state, action, w, reward, next_state, done

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class WeightSamplerAngle:
    """Sample weight vectors from normal distribution."""

    def __init__(self, rwd_dim, angle, w=None):
        """Initialize the weight sampler."""
        self.rwd_dim = rwd_dim
        self.angle = angle
        if w is None:
            w = th.ones(rwd_dim)
        w = w / th.norm(w)
        self.w = w

    def sample(self, n_sample):
        """Sample n_sample weight vectors from normal distribution."""
        s = th.normal(th.zeros(n_sample, self.rwd_dim))

        # remove fluctuation on dir w
        s = s - (s @ self.w).view(-1, 1) * self.w.view(1, -1)

        # normalize it
        s = s / th.norm(s, dim=1, keepdim=True)

        # sample angle
        s_angle = th.rand(n_sample, 1) * self.angle

        # compute shifted vector from w
        w_sample = th.tan(s_angle) * s + self.w.view(1, -1)

        w_sample = w_sample / th.norm(w_sample, dim=1, keepdim=True, p=1)

        return w_sample.float()


class Policy(nn.Module):
    """Policy network."""

    def __init__(self, obs_dim, rew_dim, output_dim, action_space, net_arch=[256, 256]):
        """Initialize the policy network."""
        super().__init__()
        self.action_space = action_space
        self.latent_pi = mlp(obs_dim + rew_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], output_dim)
        self.log_std_linear = nn.Linear(net_arch[-1], output_dim)
        self.action_dim = output_dim

        # action rescaling
        self.register_buffer("action_scale", th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32))
        self.register_buffer("action_bias", th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32))

        self.apply(layer_init)

    def forward(self, obs, w):
        """Forward pass of the policy network."""
        h = self.latent_pi(th.concat((obs, w), dim=obs.dim() - 1))
        mean = self.mean(h)
        log_std = self.log_std_linear(h)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    
    def get_action(self, summary, w):
        """Get an action from the policy network."""
        bs, seq_len = summary.shape[0], summary.shape[1]
        mean, _ = self.forward(summary, w)
        y_t = th.tanh(mean).view(bs, seq_len, self.action_dim)
        action = y_t * self.action_scale + self.action_bias
        return action
    
    def sample(
        self, 
        summary, 
        w,
    ):
        """Sample an action from the policy network."""
        bs, seq_len = summary.shape[0], summary.shape[1]  # seq_len can be 1 (inference) or num_bptt (training)
        # for each state in the mini-batch, get its mean and std
        means, log_stds = self.forward(summary, w)
        means, stds = means.view(bs * seq_len, self.action_dim), log_stds.exp().view(bs * seq_len, self.action_dim)

        # sample actions
        mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)  # normal distribution
        x_t = mu_given_s.rsample()  # for reparameterization trick (mean + std * N(0,1))

        # restrict the outputs
        y_t = th.tanh(x_t)
        action = y_t.view(bs, seq_len, self.action_dim) * self.action_scale + self.action_bias

        # Enforcing Action Bound
        # compute the log_prob as the normal distribution sample is processed by tanh
        #       (reparameterization trick)
        log_prob = mu_given_s.log_prob(x_t) - th.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=1)
        log_prob = log_prob.clamp(-1e3, 1e3)

        return action, log_prob.view(bs, seq_len, 1)

class Summarizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, recurrent_type='lstm'):
        super().__init__()

        if recurrent_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        else:
            raise ValueError(f"{recurrent_type} not recognized")

    def forward(self, observations, hidden=None, return_hidden=False):
        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(observations, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary

class QNetwork(nn.Module):
    """Q-network S x Ax W -> R^reward_dim."""

    def __init__(self, obs_dim, action_dim, rew_dim, net_arch=[256, 256]):
        """Initialize the Q-network."""
        super().__init__()
        self.net = mlp(obs_dim + action_dim + rew_dim, rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, action, w):
        """Forward pass of the Q-network."""
        q_values = self.net(th.cat((obs, action, w), dim=obs.dim() - 1))
        return q_values
    
def split_obs_from_context(obs, context_dim):
    return obs[:, :, :obs.shape[2] - context_dim]

def mean_of_unmasked_elements(tensor: th.tensor, mask: th.tensor) -> th.tensor:
    return th.mean(tensor * mask) / mask.sum() * np.prod(mask.shape)

class CAPQLRNN(RecurrentMOPolicy, MOAgent):
    """CAPQL algorithm with recurrent policy and Q-networks.
    """

    def __init__(
        self,
        env: gym.Env,
        asymmetric: bool = False,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1000000,
        net_arch: List = [256, 256],
        batch_size: int = 128,
        num_q_nets: int = 2,
        alpha: float = 0.2,
        learning_starts: int = 1000,
        gradient_updates: int = 1,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "CAPQLRNN",
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_tags: List[str] = [],
        offline_mode: bool = False,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ):
        """CAPQL algorithm with continuous actions.

        It extends the Soft-Actor Critic algorithm to multi-objective RL.
        It learns the policy and Q-networks conditioned on the weight vector.

        Args:
            env (gym.Env): The environment to train on.
            learning_rate (float, optional): The learning rate. Defaults to 3e-4.
            gamma (float, optional): The discount factor. Defaults to 0.99.
            tau (float, optional): The soft update coefficient. Defaults to 0.005.
            buffer_size (int, optional): The size of the replay buffer. Defaults to int(1e6).
            net_arch (List, optional): The network architecture for the policy and Q-networks.
            batch_size (int, optional): The batch size for training. Defaults to 256.
            num_q_nets (int, optional): The number of Q-networks to use. Defaults to 2.
            alpha (float, optional): The entropy regularization coefficient. Defaults to 0.2.
            learning_starts (int, optional): The number of steps to take before starting to train. Defaults to 100.
            gradient_updates (int, optional): The number of gradient steps to take per update. Defaults to 1.
            project_name (str, optional): The name of the project. Defaults to "MORL Baselines".
            experiment_name (str, optional): The name of the experiment. Defaults to "GPI-PD Continuous Action".
            wandb_entity (Optional[str], optional): The wandb entity. Defaults to None.
            wandb_group: The wandb group to use for logging.
            wandb_tags: Extra wandb tags to use for experiment versioning.
            offline_mode (bool, optional): Whether to run wandb in offline mode. Defaults to False.
            log (bool, optional): Whether to log to wandb. Defaults to True.
            seed (Optional[int], optional): The seed to use. Defaults to None.
            device (Union[th.device, str], optional): The device to use for training. Defaults to "auto".
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        RecurrentMOPolicy.__init__(self, device=device)
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.num_q_nets = num_q_nets
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.alpha = alpha
        self.sequence_length = self.env.spec.max_episode_steps
        self.asymmetric = asymmetric

        if asymmetric:
            assert self.context_dim is not None, "Context shape must be provided for asymmetric networks."
        else:
            self.context_dim = 0 # no context
        
        # we will split the context from the observation if asymmetric but bptt will always happen
        # because we are using a recurrent actor
        self.replay_buffer = RecurrentReplayBuffer(
            self.observation_dim + self.context_dim,
            self.action_dim, 
            self.reward_dim, 
            self.buffer_size // self.sequence_length,
            self.batch_size,
            self.sequence_length,
            device=self.device
        )

        input_dim = self.observation_dim
        if not asymmetric: # use recurrent summarizer
            self.q_summarizers = [
                Summarizer(input_dim, net_arch[0]).to(self.device)
                for _ in range(num_q_nets)
            ]
            self.target_q_summarizers = [
                Summarizer(input_dim, net_arch[0]).to(self.device)
                for _ in range(num_q_nets)
            ]
            input_dim = net_arch[0]
        else:
            input_dim = self.observation_dim + self.context_dim

        self.q_nets = [
            QNetwork(input_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]
        self.target_q_nets = [
            QNetwork(input_dim, self.action_dim, self.reward_dim, net_arch=net_arch).to(self.device)
            for _ in range(num_q_nets)
        ]

        if not asymmetric:
            for q_summarizer, target_q_summarizer in zip(self.q_summarizers, self.target_q_summarizers):
                target_q_summarizer.load_state_dict(q_summarizer.state_dict())
                for param in target_q_summarizer.parameters():
                    param.requires_grad = False

        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
            for param in target_q_net.parameters():
                param.requires_grad = False

        # actor always uses recurrent network
        self.actor_summarizer = Summarizer(self.observation_dim, net_arch[0]).to(self.device)
        self.actor = Policy(
            net_arch[0], self.reward_dim, self.action_dim, self.env.action_space, net_arch=net_arch
        ).to(self.device)

        self.actor_summarizer_optim = optim.Adam(self.actor_summarizer.parameters(), lr=self.learning_rate)
        if not asymmetric:
            self.q_summarizer_optims = optim.Adam(
                chain(*[summarizer.parameters() for summarizer in self.q_summarizers]), lr=self.learning_rate
            )
        self.q_optim = optim.Adam(chain(*[net.parameters() for net in self.q_nets]), lr=self.learning_rate)
        self.actor_optim = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)

        self._n_updates = 0

        self.log = log
        if self.log:
            self.experiment_name = experiment_name
            if asymmetric:
                self.experiment_name += f"+Asym"
            self.setup_wandb(project_name, self.experiment_name, wandb_entity, wandb_group, wandb_tags, offline_mode)

    def get_config(self):
        """Get the configuration of the agent."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "num_q_nets": self.num_q_nets,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "alpha": self.alpha,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def save(self, save_dir="weights/", filename=None, save_replay_buffer=True):
        """Save the agent's weights and replay buffer."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        saved_params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optim.state_dict(),
            "actor_summarizer_state_dict": self.actor_summarizer.state_dict(),
            "actor_summarizer_optimizer_state_dict": self.actor_summarizer_optim.state_dict(),
        }
        if not self.asymmetric:
            for i, (q_summarizer, target_q_summarizer) in enumerate(zip(self.q_summarizers, self.target_q_summarizers)):
                saved_params["q_summarizer_" + str(i) + "_state_dict"] = q_summarizer.state_dict()
                saved_params["target_q_summarizer_" + str(i) + "_state_dict"] = target_q_summarizer.state_dict()
            saved_params["q_summarizers_optimizer_state_dict"] = self.q_summarizer_optims.state_dict()
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            saved_params["q_net_" + str(i) + "_state_dict"] = q_net.state_dict()
            saved_params["target_q_net_" + str(i) + "_state_dict"] = target_q_net.state_dict()
        saved_params["q_nets_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path, load_replay_buffer=True):
        """Load the agent weights from a file."""
        params = th.load(path, map_location=self.device)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_summarizer.load_state_dict(params["actor_summarizer_state_dict"])
        self.actor_summarizer_optim.load_state_dict(params["actor_optimizer_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optimizer_state_dict"])
        if not self.asymmetric:
            for i, (q_summarizer, target_q_summarizer) in enumerate(zip(self.q_summarizers, self.target_q_summarizers)):
                q_summarizer.load_state_dict(params["q_summarizer_" + str(i) + "_state_dict"])
                target_q_summarizer.load_state_dict(params["target_q_summarizer_" + str(i) + "_state_dict"])
            self.q_summarizer_optims.load_state_dict(params["q_summarizers_optimizer_state_dict"])
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(params["q_net_" + str(i) + "_state_dict"])
            target_q_net.load_state_dict(params["target_q_net_" + str(i) + "_state_dict"])
        self.q_optim.load_state_dict(params["q_nets_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self, batch: RecurrentBatch):
        """Update the policy and the Q-nets."""
        bs, num_bptt = batch.m.size(0), batch.m.size(1)
        eps_obs = split_obs_from_context(batch.o, self.context_dim)

        actor_summary = self.actor_summarizer(eps_obs)
        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, :-1, :], actor_summary[:, 1:, :]
        assert actor_summary.shape == (bs, num_bptt+1, self.net_arch[0])

        if not self.asymmetric: # use recurrent critic
            q_summaries = [summarizer(batch.o) for summarizer in self.q_summarizers]
            target_q_summaries = [target_summarizer(batch.o) for target_summarizer in self.target_q_summarizers]
            
            q_summary_1_T, q_summary_2_Tplus1 = \
                [q_summary[:, :-1, :] for q_summary in q_summaries], \
                [target_q_summary[:, 1:, :] for target_q_summary in target_q_summaries]

            q_predictions = [q_net(q_summary_1_T[i], batch.a, batch.w) for i, q_net in enumerate(self.q_nets)]
            assert q_predictions[0].shape == (bs, num_bptt, self.reward_dim)
        else: # use asymmetric critic
            q_summary_1_T, q_summary_2_Tplus1 = batch.o[:, :-1, :], batch.o[:, 1:, :]
            assert q_summary_1_T.shape == (bs, num_bptt, self.observation_dim + self.context_dim)

            q_predictions = [q_net(q_summary_1_T, batch.a, batch.w) for q_net in self.q_nets]
            assert q_predictions[0].shape == (bs, num_bptt, self.reward_dim)


        with th.no_grad():
            next_actions, log_pi_na_given_ns = self.actor.sample(
                actor_summary_2_Tplus1, # get action for next obs
                batch.w,
            )
            if not self.asymmetric:
                q_targets = th.stack([q_target(q_summary_2_Tplus1[i], next_actions, batch.w) for i, q_target in enumerate(self.target_q_nets)])
            else:
                q_targets = th.stack([q_target(q_summary_2_Tplus1, next_actions, batch.w) for q_target in self.target_q_nets])
            
            n_min_Q_targ = th.min(q_targets, dim=0)[0]
            n_sample_entropy = -log_pi_na_given_ns

            targets = (batch.r + (1 - batch.d) * self.gamma * (n_min_Q_targ + self.alpha * n_sample_entropy)).detach()

            assert next_actions.shape == (bs, num_bptt, self.action_dim)
            assert targets.shape == (bs, num_bptt, self.reward_dim)
            assert n_min_Q_targ.shape == (bs, num_bptt, self.reward_dim)
            assert targets.shape == (bs, num_bptt, self.reward_dim)

        # compute td error
        q_losses_elementwise = [((q_pred - targets) ** 2) for q_pred in q_predictions]
        critic_loss = (1 / self.num_q_nets) * sum([mean_of_unmasked_elements(q_loss, batch.m) for q_loss in q_losses_elementwise])

        assert critic_loss.shape == ()
        if not self.asymmetric:
            self.q_summarizer_optims.zero_grad()
        self.q_optim.zero_grad()
        critic_loss.backward()
        if not self.asymmetric:
            self.q_summarizer_optims.step()
        self.q_optim.step()

        # Policy update
        actions, log_pi_a_given_ns = self.actor.sample(
            actor_summary_1_T, # get action for current obs
            batch.w, 
        )
        if not self.asymmetric:
            q_values = th.stack([q_net(q_summary_1_T[i], actions, batch.w) for i, q_net in enumerate(self.q_nets)])
        else:
            q_values = th.stack([q_net(q_summary_1_T, actions, batch.w) for q_net in self.q_nets])
        min_Q = th.min(q_values.detach(), dim=0)[0] # detach from the computation graph for critic loss
        min_Q = (min_Q * batch.w).sum(dim=-1, keepdim=True) # weighted sum
        policy_loss_elementwise = -(min_Q + self.alpha * -log_pi_a_given_ns)
        policy_loss = mean_of_unmasked_elements(policy_loss_elementwise, batch.m)

        assert actions.shape == (bs, num_bptt, self.action_dim)
        assert log_pi_a_given_ns.shape == (bs, num_bptt, 1)
        assert min_Q.shape == (bs, num_bptt, 1)
        assert policy_loss.shape == ()

        # reduce policy loss
        self.actor_summarizer_optim.zero_grad()
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_summarizer_optim.step()
        self.actor_optim.step()

        if not self.asymmetric:
            for q_summarizer, target_q_summarizer in zip(self.q_summarizers, self.target_q_summarizers):
                polyak_update(q_summarizer.parameters(), target_q_summarizer.parameters(), self.tau)

        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            polyak_update(q_net.parameters(), target_q_net.parameters(), self.tau)

        if self.log and self.global_step % 100 == 0:
            wandb.log(
                {
                    "losses/critic_loss": critic_loss.item(),
                    "losses/policy_loss": policy_loss.item(),
                    "global_step": self.global_step,
                },
            )
    @th.no_grad()
    def act(self, observation: th.Tensor, w: th.Tensor, deterministic: bool) -> np.array:
        # make sure the observation is of shape (1, 1, obs_dim) because training is done with (bs, num_bptt, obs_dim)
        observation = observation.unsqueeze(0).unsqueeze(0)
        w = w.unsqueeze(0).unsqueeze(0)
        if self.asymmetric:
            observation = split_obs_from_context(observation, self.context_dim)
        summary, self.hidden = self.actor_summarizer(observation, self.hidden, return_hidden=True)
        if deterministic:
            action = self.actor.get_action(summary, w)
        else:
            action = self.actor.sample(summary, w)
        return action.view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy

    @th.no_grad()
    def eval(
        self, 
        obs: Union[np.ndarray, th.Tensor],
        w: Union[np.ndarray, th.Tensor], 
        torch_action=False,
        num_envs: int = 1,
        **kwargs
    ) -> Union[np.ndarray, th.Tensor]:
        """Evaluate the policy action for the given observation and weight vector."""
        if isinstance(obs, np.ndarray):
            obs = th.tensor(obs).float().to(self.device)
            w = th.tensor(w).float().to(self.device)

        if num_envs == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
            w = w.unsqueeze(0).unsqueeze(0)
        else: # (num_envs, obs_dim) -> (num_envs, 1, obs_dim)
            obs = obs.unsqueeze(1)
            w = w.unsqueeze(1)
        summary, self.hidden = self.actor_summarizer(obs, self.hidden, return_hidden=True)
        action = self.actor.get_action(summary, w).squeeze(1)

        if not torch_action:
            action = action.detach().cpu().numpy()
        
        return action
    
    def copy_networks_from(self, algorithm) -> None:
        self.actor_summarizer.load_state_dict(algorithm.actor_summarizer.state_dict())
        self.actor.load_state_dict(algorithm.actor.state_dict())
        if not self.asymmetric:
            for i, (q_summarizer, target_q_summarizer) in enumerate(zip(self.q_summarizers, self.target_q_summarizers)):
                q_summarizer.load_state_dict(algorithm.q_summarizers[i].state_dict())
                target_q_summarizer.load_state_dict(algorithm.target_q_summarizers[i].state_dict())
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            q_net.load_state_dict(algorithm.q_nets[i].state_dict())
            target_q_net.load_state_dict(algorithm.target_q_nets[i].state_dict())
    
    def get_networks(self) -> dict:
        networks = {
            "actor_summarizer": deepcopy(self.actor_summarizer),
            "actor": deepcopy(self.actor),
        }
        if not self.asymmetric:
            for i, (q_summarizer, target_q_summarizer) in enumerate(zip(self.q_summarizers, self.target_q_summarizers)):
                networks["q_summarizer_" + str(i)] = deepcopy(q_summarizer)
                networks["target_q_summarizer_" + str(i)] = deepcopy(target_q_summarizer)
        for i, (q_net, target_q_net) in enumerate(zip(self.q_nets, self.target_q_nets)):
            networks["q_net_" + str(i)] = deepcopy(q_net)
            networks["target_q_net_" + str(i)] = deepcopy(target_q_net)
        
        return networks

    def train(
        self,
        total_timesteps: int,
        eval_env: Union[gymnasium.Env, MORLGeneralizationEvaluator],
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        eval_mo_freq: int = 10000,
        reset_num_timesteps: bool = False,
        checkpoints: bool = False,
        verbose: bool = False,
        test_generalization: bool = False,
    ):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps to train the agent for.
            eval_env (gym.Env): Environment to use for evaluation.
            ref_point (np.ndarray): Reference point for hypervolume calculation.
            known_pareto_front (Optional[List[np.ndarray]]): Optimal Pareto front, if known.
            num_eval_weights_for_front (int): Number of weights to evaluate for the Pareto front.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            eval_mo_freq (int): Number of timesteps between evaluations during an iteration.
            reset_num_timesteps (bool): Whether to reset the number of timesteps.
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
                    "eval_mo_freq": eval_mo_freq,
                    "reset_num_timesteps": reset_num_timesteps,
                }
            )

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        angle = th.pi * (22.5 / 180)
        weight_sampler = WeightSamplerAngle(self.env.unwrapped.reward_dim, angle)

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, info = self.env.reset()


        # Since algorithm is a recurrent policy, it (ideally) shouldn't be updated during an episode since this would
        # affect its ability to interpret past hidden states. Therefore, during an episode, algorithm_clone is updated
        # while algorithm is not. Once an episode has finished, we do algorithm.copy_networks_from(algorithm_clone) to
        # carry over the changes.
        algorithms_clone = deepcopy(self)  # algorithms_clone is for updates

        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            tensor_w = weight_sampler.sample(1).view(-1).to(self.device)
            w = tensor_w.detach().cpu().numpy()

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                with th.no_grad():
                    action = self.act(
                        th.tensor(obs).float().to(self.device),
                        tensor_w,
                        deterministic=True,
                    )

            action_env = action

            next_obs, vector_reward, terminated, truncated, info = self.env.step(action_env)

            self.replay_buffer.push(obs, action, vector_reward, w, next_obs, terminated, truncated)

            if terminated or truncated:
                obs, info = self.env.reset()
                self.num_episodes += 1

                self.copy_networks_from(algorithms_clone)
                self.zero_start_rnn_hidden() # crucial for recurrent policy

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step, verbose=verbose)
            else:
                obs = next_obs

            if self.global_step >= self.learning_starts:
                batch = self.replay_buffer.sample()
                algorithms_clone.update(batch)


            if self.log and self.global_step % eval_mo_freq == 0:
                # Evaluation
                test_algo = deepcopy(algorithms_clone)
                test_algo.zero_start_rnn_hidden()
                if test_generalization:
                    eval_env.eval(test_algo, ref_point=ref_point, global_step=self.global_step)
                else:
                    returns_test_tasks = [
                        policy_evaluation_mo(test_algo, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                    ]
                    log_all_multi_policy_metrics(
                        current_front=returns_test_tasks,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

            # Checkpoint
            if checkpoints:
                self.save(filename="CAPQL", save_replay_buffer=False)

        if self.log:
            self.close_wandb()
