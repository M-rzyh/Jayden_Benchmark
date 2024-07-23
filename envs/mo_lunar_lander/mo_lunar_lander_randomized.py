import os
import gymnasium as gym
import torch
import numpy as np
from .utils.mo_lunar_lander import MOLunarLander
from envs.registration import register as gym_register
from envs.generalization_evaluator import DREnv

def rand_int_seed():
    return int.from_bytes(os.urandom(4), byteorder="little")

class MOLunarLanderDR(MOLunarLander, DREnv):
    param_info = {'names': ['gravity', 'wind_power'],
                  'param_max': [0.0, 20.0],
                  'param_min': [-12.0, 0.0]
                }
    DEFAULT_PARAMS = [-10.0,15.0]

    def __init__(self, seed = 0, random_z_dim = 10, continuous = True):
        DREnv.__init__(self)
        MOLunarLander.__init__(self, enable_wind=True, continuous=continuous)

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
        return super().reset()

    def reset_random(self):
        """
        Reset the environment with a new parameters. Please call self.reset() after this.
        """
        params_max = self.param_info['param_max']
        params_min = self.param_info['param_min']
        new_params = [
            np.random.uniform(params_min[0], params_max[0]),
            np.random.uniform(params_min[1], params_max[1])
        ]
        self._update_params(*new_params)
        # return self.reset()

    def step_adversary(self, action):
        param_max = self.param_info['param_max'][self.adversary_step_count]
        param_min = self.param_info['param_min'][self.adversary_step_count]
        if torch.is_tensor(action):
            action = action.item()

        value = ((action + 1)/2) * (param_max - param_min) + param_min
        self.level_params_vec[self.adversary_step_count] = value

        self.adversary_step_count += 1

        if self.adversary_step_count >= self.adversary_max_steps:
            self._update_params(*self.level_params_vec)
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
    
    def _update_params(self, gravity, wind_power):
        self.wind_power = wind_power
        self.gravity = gravity

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def reset_to_level(self, level, seed):
        if isinstance(level, str):
            encoding = list(np.fromstring(level))
        else:
            encoding = [float(x) for x in level[:-1]]

        assert len(level) == len(self.level_params_vec), \
            'Level input is the wrong length.'

        self.level_params_vec = encoding
        self._update_params(*self.level_params_vec)

        return super().reset(seed=seed)

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

