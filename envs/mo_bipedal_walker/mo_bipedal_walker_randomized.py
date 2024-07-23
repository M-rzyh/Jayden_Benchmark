# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import gymnasium as gym
import numpy as np
import torch

from envs.mo_bipedal_walker.utils.mo_bipedal_walker import MOBipedalWalker
from envs.generalization_evaluator import DREnv
from collections import namedtuple

EnvConfig = namedtuple('EnvConfig', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

PARAM_RANGES_DEBUG = {
    1: [0,0.01], # ground roughness
    2: [0,0], # pit gap 1
    3: [0.01,0.01], # pit gap 2
    4: [0,0], # stump height 1
    5: [0.01,0.01], # stump height 2
    6: [0,0], # stair height 1
    7: [0.01,0.01], # stair height 2
    8: [1,1], # stair steps
}

PARAM_RANGES_EASY = {
    1: [0,0.6], # ground roughness
    2: [0,0], # pit gap 1
    3: [0.8,0.8], # pit gap 2
    4: [0,0], # stump height 1
    5: [0.4,0.4], # stump height 2
    6: [0,0], # stair height 1
    7: [0.4,0.4], # stair height 2
    8: [1,1], # stair steps
}

PARAM_RANGES_FULL = {
    1: [0,10], # ground roughness
    2: [0,10], # pit gap 1
    3: [0,10], # pit gap 2
    4: [0,5], # stump height 1
    5: [0,5], # stump height 2
    6: [0,5], # stair height 1
    7: [0,5], # stair height 2
    8: [1,9], # stair steps
}

PARAM_MUTATIONS = {
    1: [0,0.6], # ground roughness
    2: [0.4], # pit gap 1
    3: [0.4], # pit gap 2
    4: [0.2], # stump height 1
    5: [0.2], # stump height 2
    6: [0.2], # stair height 1
    7: [0.2], # stair height 2
    8: [1], # stair steps
}

DEFAULT_LEVEL_PARAMS_VEC = [0,0,10,0,5,0,5,9]
STUMP_WIDTH_RANGE = [1, 2]
STUMP_FLOAT_RANGE = [0, 1]
STAIR_WIDTH_RANGE = [4, 5]


def rand_int_seed():
    return int.from_bytes(os.urandom(4), byteorder="little")


class MOBipedalWalkerDR(MOBipedalWalker, DREnv):
    def __init__(self, mode='full', poet=False, random_z_dim=10, seed=0):
        DREnv.__init__(self)
        self.mode = mode
        self.level_seed = seed
        self.poet = poet # POET didn't use the stairs, not clear why

        default_config = EnvConfig(
            name='default_conf',
            ground_roughness=0,
            pit_gap=[0,10],
            stump_width=[4,5],
            stump_height=[0,5],
            stump_float=[0,1],
            stair_height=[0,5],
            stair_width=[4,5],
            stair_steps=[1])

        super().__init__(default_config)

        if self.poet:
            self.adversary_max_steps = 5
        else:
            self.adversary_max_steps = 8
        self.random_z_dim = random_z_dim
        self.passable = True

        # Level vec is the *tunable* DR params
        self.level_params_vec = DEFAULT_LEVEL_PARAMS_VEC
        if self.poet:
            self.level_params_vec = self.level_params_vec[:5]
        self._update_params(self.level_params_vec)

        if poet:
            self.mutations = {k:v for k,v in list(PARAM_MUTATIONS.items())[:5]}
        else:
            self.mutations = PARAM_MUTATIONS

        n_u_chars = max(12, len(str(rand_int_seed())))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))

        # Fixed params
        self.stump_width = STUMP_WIDTH_RANGE
        self.stump_float = STUMP_FLOAT_RANGE
        self.stair_width = STAIR_WIDTH_RANGE

        # Create spaces for adversary agent's specs.
        self.adversary_action_dim = 1
        self.adversary_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.adversary_ts_obs_space = \
            gym.spaces.Box(
                low=0,
                high=self.adversary_max_steps,
                shape=(1,),
                dtype='uint8')
        self.adversary_randomz_obs_space = \
            gym.spaces.Box(
                low=0,
                high=1.0,
                shape=(random_z_dim,),
                dtype=np.float32)
        self.adversary_image_obs_space = \
            gym.spaces.Box(
                low=0,
                high=10.0,
                shape=(len(self.level_params_vec),),
                dtype=np.float32)
        self.adversary_observation_space = \
            gym.spaces.Dict({
                'image': self.adversary_image_obs_space,
                'time_step': self.adversary_ts_obs_space,
                'random_z': self.adversary_randomz_obs_space})

    # def reset(self):
    #     self.step_count = 0
    #     self.adversary_step_count = 0

    #     # Reset to default parameters
    #     self.level_params_vec = DEFAULT_LEVEL_PARAMS_VEC
    #     if self.poet:
    #         self.level_params_vec = self.level_params_vec[:5]

    #     self._update_params(self.level_params_vec)

    #     self.level_seed = rand_int_seed()

    #     obs = {
    #         'image': self.get_obs(),
    #         'time_step': [self.adversary_step_count],
    #         'random_z': self.generate_random_z()
    #     }

    #     return obs

    def get_obs(self):
        ## vector of *tunable* environment params
        obs = []
        obs += [self.ground_roughness]
        obs += self.pit_gap
        obs += self.stump_height
        if not self.poet:
            obs += self.stair_height
            obs += self.stair_steps

        return np.array(obs)

    def reset_agent(self):
        return super().reset()

    def _update_params(self, level_params_vec):
        self.ground_roughness = level_params_vec[0]
        self.pit_gap = [level_params_vec[1],level_params_vec[2]]
        self.pit_gap.sort()
        self.stump_height = [level_params_vec[3],level_params_vec[4]]
        self.stump_height.sort()
        if self.poet:
            self.stair_height = []
            self.stair_steps = []
        else:
            self.stair_height = [level_params_vec[5],level_params_vec[6]]
            self.stair_height.sort()
            self.stair_steps = [int(round(level_params_vec[7]))]

    def get_complexity_info(self):
        complexity_info = {
            'ground_roughness': self.ground_roughness,
            'pit_gap_low': self.pit_gap[0],
            'pit_gap_high': self.pit_gap[1],
            'stump_height_low': self.stump_height[0],
            'stump_height_high': self.stump_height[1]
        }

        if not self.poet:
            complexity_info['stair_height_low'] = self.stair_height[0]
            complexity_info['stair_height_high'] = self.stair_height[1]
            complexity_info['stair_steps'] = self.stair_steps[0]

        return complexity_info

    def get_config(self):
        """
        Gets the config to use to create the level.
        If the range is zer or below a min threshold, we put blank entries.
        """
        if self.stump_height[1] < 0.2:
            stump_height = []
            stump_width = []
            stump_float = []
        else:
            stump_height = self.stump_height
            stump_width = self.stump_width
            stump_float = self.stump_float

        if self.pit_gap[1] < 0.8:
            pit_gap = []
        else:
            pit_gap = self.pit_gap

        if self.poet:
            stair_height = []
            stair_width = []
            stair_steps = []
        elif self.stair_height[1] < 0.2:
            stair_height = []
            stair_width = []
            stair_steps = []
        else:
            stair_height = self.stair_height
            stair_width = self.stair_width
            stair_steps = self.stair_steps

        # get the current config
        config = EnvConfig(
            name='config',
            ground_roughness=self.ground_roughness,
            pit_gap=pit_gap,
            stump_width=stump_width,
            stump_height=stump_height,
            stump_float=stump_float,
            stair_height=stair_height,
            stair_width=stair_width,
            stair_steps=stair_steps)

        return config

    def _reset_env_config(self):
        """
        Resets the environment based on current level encoding.
        """
        config = self.get_config()
        super().re_init(config)

    def reset_to_level(self, level):
        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]] + [int(level[-1])]

        assert len(level) == len(self.level_params_vec) + 1, \
            f'Level input is the wrong length.'

        self.level_params_vec = encoding[:-1]
        self._update_params(self.level_params_vec)
        self._reset_env_config()

        self.level_seed = int(level[-1])

        return super().reset(seed=self.level_seed)

    @property
    def param_ranges(self):
        if self.mode == 'easy':
            param_ranges = PARAM_RANGES_EASY
        elif self.mode == 'full':
            param_ranges = PARAM_RANGES_FULL
        elif self.mode == 'debug':
            param_ranges = PARAM_RANGES_DEBUG
        else:
            raise ValueError("Mode must be 'easy' or 'full'")

        return param_ranges

    @property
    def encoding(self):
        enc = self.level_params_vec + [self.level_seed]
        enc = [str(x) for x in enc]
        return np.array(enc, dtype=self.encoding_u_chars)

    @property
    def level(self):
        return self.encoding

    def reset_random(self):
        param_ranges = self.param_ranges

        rand_norm_params = np.random.rand(len(param_ranges))
        self.level_params_vec = \
            [rand_norm_params[i]*(param_range[1]-param_range[0]) + param_range[0]
                for i,param_range in enumerate(param_ranges.values())]
        self._update_params(self.level_params_vec)

        self.level_seed = rand_int_seed()

        self._reset_env_config()
        # return super().reset(seed=self.level_seed)

    @property
    def processed_action_dim(self):
        return 1

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def mutate_level(self, num_edits=1):
        if num_edits > 0:
            # Perform mutations on current level vector
            param_ranges = self.param_ranges
            edit_actions = np.random.randint(1, len(self.mutations) + 1, num_edits)
            edit_dirs = np.random.randint(0, 3, num_edits) - 1

            # Update level_params_vec
            for a,d in zip(edit_actions, edit_dirs):
                mutation_range = self.mutations[a]
                if len(mutation_range) == 1:
                    mutation = d*mutation_range[0]
                elif len(mutation_range) == 2:
                    mutation = d*np.random.uniform(*mutation_range)

                self.level_params_vec[a-1] = \
                    np.clip(self.level_params_vec[a-1]+mutation,
                            *PARAM_RANGES_FULL[a])

            self.level_seed = rand_int_seed()
            self._update_params(self.level_params_vec)
            self._reset_env_config()

        return super().reset(seed=self.level_seed)

    def step_adversary(self, action):
        # action will be between [-1,1]
        # this maps to a range, depending on the index
        param_ranges = self.param_ranges
        val_range = param_ranges[self.adversary_step_count+1]

        if torch.is_tensor(action):
            action = action.item()

        # get unnormalized value from the action
        value = ((action + 1)/2) * (val_range[1]-val_range[0]) + val_range[0]

        # update the level vec
        self.level_params_vec[self.adversary_step_count] = value

        self.adversary_step_count += 1

        if self.adversary_step_count >= self.adversary_max_steps:
            self.level_seed = rand_int_seed()
            self._update_params(self.level_params_vec)
            self._reset_env_config()
            done=True
        else:
            done=False

        obs = {
            'image': self.level_params_vec,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs, 0, done, {}


class BipedalWalkerFull(MOBipedalWalkerDR):
  def __init__(self, seed=0):
    super().__init__(mode='full', seed=seed)

class BipedalWalkerEasy(MOBipedalWalkerDR):
  def __init__(self, seed=0):
    super().__init__(mode='easy', seed=seed)

class BipedalWalkerPOET(MOBipedalWalkerDR):
  def __init__(self, seed=0):
    super().__init__(mode='full', poet=True, seed=seed)

class BipedalWalkerEasyPOET(MOBipedalWalkerDR):
  def __init__(self, seed=0):
    super().__init__(mode='easy', poet=True, seed=seed)
