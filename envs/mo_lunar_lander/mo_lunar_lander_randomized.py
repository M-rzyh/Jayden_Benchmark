import os
import gymnasium as gym
import numpy as np
from .utils.mo_lunar_lander import MOLunarLander
from morl_generalization.algos.dr import DREnv
from gymnasium.utils import EzPickle

class MOLunarLanderDR(MOLunarLander, DREnv, EzPickle):
    param_info = {'names': ['gravity', 'wind_power', 'turbulence_power'],
                  'param_max': [0.0, 20.0, 2.0],
                  'param_min': [-15.0, 0.0, 0.0]
                }
    DEFAULT_PARAMS = [-10.0, 15.0, 1.5]

    def __init__(self, continuous=False, enable_wind=True, **kwargs):
        DREnv.__init__(self)
        EzPickle.__init__(self,
            continuous,
            enable_wind,
            **kwargs,
        )
        MOLunarLander.__init__(self, enable_wind=enable_wind, continuous=continuous, **kwargs)

    def reset_random(self):
        """
        Reset the environment with a new parameters. Please call self.reset() after this.
        """
        params_max = self.param_info['param_max']
        params_min = self.param_info['param_min']
        new_params = [
            np.random.uniform(params_min[0], params_max[0]),
            np.random.uniform(params_min[1], params_max[1]),
            np.random.uniform(params_min[2], params_max[2]),
        ]
        self._update_params(*new_params)

    def _update_params(self, gravity, wind_power, turbulence_power):
        self.wind_power = wind_power
        self.gravity = gravity
        self.turbulence_power = turbulence_power

    def get_complexity_info(self):
        info = {
            'gravity': self.gravity,
            'wind_power': self.wind_power,
            'turbulence_power': self.turbulence_power
        }
        return info
