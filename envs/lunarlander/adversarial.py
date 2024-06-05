import os
import gym
import torch
import numpy as np
from .lunarlander_parameterized import LunarLander
from envs.registration import register as gym_register

def rand_int_seed():
    return int.from_bytes(os.urandom(4), byteorder="little")

class LunarLanderAdversarial(LunarLander):
    param_info = {'names': ['gravity', 'wind_power'],
                  'param_max':np.array([0.0, 20.0]),
                  'param_min': np.array([-12.0, 0.0])
                }
    DEFAULT_PARAMS = [-10.0,15.0]

    def __init__(self, seed = 0, random_z_dim = 10):
        super().__init__(enable_wind=True)

        self.passable = True
        self.level_seed = seed
        self.random_z_dim = random_z_dim
        self.level_params_vec = self.DEFAULT_PARAMS
        self.adversary_step_count = 0
        self.adversary_max_steps = len(self.param_info['names'])
        self.adversary_action_dim = 1
        self.adversary_action_space = gym.spaces.Box(low = -1, high = 1, shape = (1,), dtype = np.float32)

        n_u_chars = max(12, len(str(rand_int_seed())))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))

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
                low=np.array([-12.0, 0.0]),
                high=np.array([0.0, 20.0]),
                shape=(len(self.level_params_vec),),
                dtype=np.float32)
        self.adversary_observation_space = \
            gym.spaces.Dict({
                'image': self.adversary_image_obs_space,
                'time_step': self.adversary_ts_obs_space,
                'random_z': self.adversary_randomz_obs_space})

    def reset_agent(self):
        return super()._reset()

    def reset_random(self):
        params_max = self.param_info['param_max'][0]
        params_min = self.param_info['param_min'][0]
        params = [
            np.random.uniform(params_min[i], params_max[i])
            for i in enumerate(params_max)
        ]
        self.set_params(*params)
        return self._reset()

    def reset(self):
        self.adversary_step_count = 0
        self.level_params_vec = self.DEFAULT_PARAMS
        self.set_params(*self.level_params_vec)

        obs = {
            'image': self.get_obs(),
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs

    def step_adversary(self, action):
        param_max = self.param_info['param_max'][self.adversary_step_count]
        param_min = self.param_info['param_min'][self.adversary_step_count]
        if torch.is_tensor(action):
            action = action.item()

        value = ((action + 1)/2) * (param_max - param_min) + param_min
        self.level_params_vec[self.adversary_step_count] = value

        self.adversary_step_count += 1

        if self.adversary_step_count >= self.adversary_max_steps:
            self.set_params(*self.level_params_vec)
            done = True
        else:
            done = False

        obs = {
            'image': self.get_obs(),
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }
        return obs, 0, done, {}

    def get_obs(self):
        return np.array([self.wind_power, self.gravity])

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]]

        assert len(level) == len(self.level_params_vec), \
            'Level input is the wrong length.'

        self.level_params_vec = encoding
        self.set_param(*self.level_params_vec)

        return self._reset()

    def get_complexity_info(self):
        info = {
            'gravity': self.gravity,
            'wind_power': self.wind_power
        }
        return info

    @property
    def processed_action_dim(self):
        return 1

    @property
    def encoding(self):
        enc = self.level_params_vec
        enc = [str(x) for x in enc]
        return np.array(enc, dtype=self.encoding_u_chars)


class LunarLanderPERM(LunarLander):
    param_info = {'names': ['gravity', 'wind_power'],
                  'param_max':np.array([0.0, 20.0]),
                  'param_min': np.array([-12.0, 0.0])
                }
    DEFAULT_PARAMS = [-10.0,15.0]
    DEFAULT_ABILITY = [0.0]

    def __init__(self, seed = 0, random_z_dim = 10):
        super().__init__(enable_wind=True)

        self.passable = True
        self.level_seed = seed
        self.level_params_vec = self.DEFAULT_PARAMS
        self.adversary_step_count = 0
        self.adversary_max_steps = len(self.param_info['names'])
        self.adversary_action_dim = 1
        self.adversary_action_space = gym.spaces.Box(low = -1, high = 1, shape = (1,), dtype = np.float32)
        self.ability = self.DEFAULT_ABILITY

        n_u_chars = max(12, len(str(rand_int_seed())))
        self.encoding_u_chars = np.dtype(('U', n_u_chars))

        self.adversary_ts_obs_space = \
            gym.spaces.Box(
                low=0,
                high=self.adversary_max_steps,
                shape=(1,),
                dtype='uint8')
        self.adversary_ability_obs_space = \
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.DEFAULT_ABILITY),),
                dtype=np.float32)
        self.adversary_image_obs_space = \
            gym.spaces.Box(
                low=-12.0,
                high=20.0,
                shape=(len(self.level_params_vec),),
                dtype=np.float32)
        self.adversary_observation_space = \
            gym.spaces.Dict({
                'image': self.adversary_image_obs_space,
                'time_step': self.adversary_ts_obs_space,
                'random_z': self.adversary_ability_obs_space})

    def reset_agent(self):
        return super()._reset()

    def reset_random(self):
        params_max = self.param_info['param_max'][0]
        params_min = self.param_info['param_min'][0]
        params = [
            np.random.uniform(params_min[i], params_max[i])
            for i in enumerate(params_max)
        ]
        self.set_params(*params)
        return self._reset()

    def reset(self):
        self.adversary_step_count = 0
        self.level_params_vec = self.DEFAULT_PARAMS
        self.set_params(*self.level_params_vec)

        obs = {
            'image': self.get_obs(),
            'time_step': [self.adversary_step_count],
            'random_z': self.get_ability()
        }

        return obs

    def step_adversary(self, action):
        param_max = self.param_info['param_max'][self.adversary_step_count]
        param_min = self.param_info['param_min'][self.adversary_step_count]
        if torch.is_tensor(action):
            action = action.item()

        value = ((action + 1)/2) * (param_max - param_min) + param_min
        self.level_params_vec[self.adversary_step_count] = value

        self.adversary_step_count += 1

        if self.adversary_step_count >= self.adversary_max_steps:
            self.set_params(*self.level_params_vec)
            done = True
        else:
            done = False

        obs = {
            'image': self.get_obs(),
            'time_step': [self.adversary_step_count],
            'random_z': self.get_ability()
        }
        return obs, 0, done, {}

    def get_obs(self):
        return np.array([self.gravity, self.wind_power])

    def get_ability(self):
        return np.array(self.ability).reshape(-1,)

    def update_ability(self, ability):
        self.ability = ability
        return None

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def reset_to_level(self, level):
        self.reset()

        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]]

        assert len(level) == len(self.level_params_vec), \
            'Level input is the wrong length.'

        self.level_params_vec = encoding
        self.set_param(*self.level_params_vec)

        return self._reset()

    def get_complexity_info(self):
        info = {
            'gravity': self.gravity,
            'wind_power': self.wind_power
        }
        return info

    @property
    def processed_action_dim(self):
        return 1

    @property
    def encoding(self):
        enc = self.level_params_vec
        enc = [str(x) for x in enc]
        return np.array(enc, dtype=self.encoding_u_chars)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

gym_register(id='LunarLander-Adversarial-v0',
             entry_point=module_path + ':LunarLanderAdversarial',
             max_episode_steps=500)

gym_register(id='LunarLander-PERM-v0',
             entry_point=module_path + ':LunarLanderPERM',
             max_episode_steps=500)
