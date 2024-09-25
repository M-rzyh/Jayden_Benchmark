from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.spaces import Dict
import numpy as np

# TODO: implement this for all dr envs
class DREnv(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset_random(self):
        pass

    @abstractmethod
    def get_task(self):
        pass

class DRWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wrapper for DR environment."""

    def __init__(self, env: gym.Env):
        """Initialize the :class:`DRWrapper` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, *, seed=None, options=None):
        self.env.unwrapped.reset_random() # domain randomization
        
        return self.env.reset(seed=seed, options=options) 
    
class DynamicsInObs(DRWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, dynamics_mask=None):
        """
            Stack the current env dynamics to the env observation vector

            dynamics_mask: list of int
                           indices of dynamics to randomize, i.e. to condition the network on
        """
        if not isinstance(env.unwrapped, DREnv):
            raise TypeError("The environment must implement be a DREnv, i.e. implement `get_task()`, before applying DynamicsInObs.")
        gym.utils.RecordConstructorArgs.__init__(self, dynamics_mask=dynamics_mask)
        DRWrapper.__init__(self, env)

        if dynamics_mask is not None:
            self.dynamics_mask = np.array(dynamics_mask)
            task_dim = env.get_task()[self.dynamics_mask].shape[0]
        else:  # All dynamics are used
            task_dim = env.get_task().shape[0]
            self.dynamics_mask = np.arange(task_dim)

        obs_space = env.observation_space
        # low = np.concatenate([obs_space.low.flatten(), np.repeat(-np.inf, task_dim)], axis=0)
        # high = np.concatenate([obs_space.high.flatten(), np.repeat(np.inf, task_dim)], axis=0)
        # self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

        self.observation_space = Dict({
            "observation": gym.spaces.Box(low=obs_space.low, high=obs_space.high, dtype=obs_space.dtype),
            "context": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(task_dim,), dtype=obs_space.dtype),
        })

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dynamics = self.env.get_task()[self.dynamics_mask]
        obs = np.concatenate([obs.flatten(), dynamics], axis=0)
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        dynamics = self.env.get_task()[self.dynamics_mask]
        obs = np.concatenate([obs.flatten(), dynamics], axis=0)
        return obs, info