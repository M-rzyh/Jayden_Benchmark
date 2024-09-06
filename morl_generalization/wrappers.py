import os
import numpy as np
from typing import Callable, Optional

import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers.monitoring import video_recorder
from gymnasium.wrappers.frame_stack import FrameStack

from morl_generalization.algos.dr import DREnv

def make_history_informed_environment(env: gym.Env, args):
    """Wrap env
        
        :param args.stack_history: int
                             number of previous obs and actions 
        :param args.rand_only: List[int]
                               dyn param indices mask
        :param args.dyn_in_obs: bool
                                condition the policy on the true dyn params
    """

    if args.stack_history is not None:
        env = FrameStack(env, args.stack_history+1)  # FrameStack considers the current obs as 1 stack
        # env = ObsToNumpy(env)
        env = FlattenObservation(env)
        env = ActionHistoryWrapper(env, args.stack_history)

    if args.dyn_in_obs:
        env = DynamicsInObs(env, dynamics_mask=args.rand_only)

    return env

class ActionHistoryWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, history_len: int, valid_dim=False):
        """
            Augments the observation with
            a stack of the previous "history_len" actions
            taken.

            valid_dim : bool
                        if False, at the beginning of the episode, zero-valued actions
                            are used.
                        if True, an additional binary valid code is used as input to indicate whether
                        previous actions are valid or not (beginning of the episode).
        """
        gym.utils.RecordConstructorArgs.__init__(self, history_len=history_len, valid_dim=valid_dim)
        gym.Wrapper.__init__(self, env)
        assert env.action_space.sample().ndim == 1, 'Actions are assumed to be flat on one-dim vector'
        assert valid_dim == False, 'valid encoding has not been implemented yet.'

        self.history_len = history_len
        self.actions_buffer = np.zeros((history_len, env.action_space.shape[0]), dtype=np.float32)

        # Modify the observation space to include the history buffer
        obs_space = env.observation_space
        action_stack_low = np.repeat(env.action_space.low, history_len)
        action_stack_high = np.repeat(env.action_space.high, history_len)
        low = np.concatenate([obs_space.low.flatten(), action_stack_low], axis=0)
        high = np.concatenate([obs_space.high.flatten(), action_stack_high], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.actions_buffer.fill(0)
        return self._stack_actions_to_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.actions_buffer[:-1] = self.actions_buffer[1:]
        self.actions_buffer[-1] = action
        obs = self._stack_actions_to_obs(obs)
        return obs, reward, done, info

    def _stack_actions_to_obs(self, obs):
        obs = np.concatenate([obs.flatten(), self.actions_buffer.flatten()], axis=0)
        return obs
    
class DynamicsInObs(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, dynamics_mask=None):
        """
            Stack the current env dynamics to the env observation vector

            dynamics_mask: list of int
                           indices of dynamics to randomize, i.e. to condition the network on
        """
        if not isinstance(env, DREnv):
            raise TypeError("The environment must implement be a DREnv, i.e. implement `get_task()`, before applying DynamicsInObs.")
        gym.utils.RecordConstructorArgs.__init__(self, dynamics_mask=dynamics_mask)
        gym.ObservationWrapper.__init__(env)

        if dynamics_mask is not None:
            self.dynamics_mask = np.array(dynamics_mask)
            task_dim = env.get_task()[self.dynamics_mask].shape[0]
        else:  # All dynamics are used
            task_dim = env.get_task().shape[0]
            self.dynamics_mask = np.arange(task_dim)

        # self.nominal_values = env.get_task()[self.dynamics_mask].copy()  # used for normalizing dynamics values

        obs_space = env.observation_space
        low = np.concatenate([obs_space.low.flatten(), np.repeat(-np.inf, task_dim)], axis=0)
        high = np.concatenate([obs_space.high.flatten(), np.repeat(np.inf, task_dim)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def observation(self, obs):
        # norm_dynamics = self.env.get_task()[self.dynamics_mask] - self.nominal_values
        norm_dynamics = self.env.get_task()[self.dynamics_mask]
        obs = np.concatenate([obs.flatten(), norm_dynamics], axis=0)
        return obs
    
def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

    
class MORecordVideo(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Multi-objective RecordVideo wrapper for recording videos. 
    Allows intermittent recording of videos based on number of weights evaluted by specifying ``weight_trigger``.
    To increased weight_number, call `env.reset(options={"weights": w, "step":s})` at the beginning of each evaluation. 
    If weight trigger is activated, the video recorded file name will include the  current step `s` and evaluated weight `w` as a suffix. 
    `w` must be a numpy array and `s` must be an integer.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        weight_trigger: Callable[[int], bool] = None,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "mo-rl-video",
        disable_logger: bool = False,
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            weight_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this weight evaluation
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
            disable_logger (bool): Whether to disable moviepy logger or not.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            weight_trigger=weight_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )
        if episode_trigger is None and step_trigger is None and weight_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger, weight_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.weight_trigger = weight_trigger
        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder: Optional[video_recorder.VideoRecorder] = None
        self.disable_logger = disable_logger

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.episode_id = 0

        # Custom multi-objective attributes
        if self.weight_trigger:
            self.weight_id = -1
            self.current_weight = None
            self.current_step = 0

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

    def reset(self, **kwargs):
        """Reset the environment, set multi-objective weights if provided, and start video recording if enabled."""
        # Check for multi-objective weights in kwargs
        options = kwargs.get("options", {})
        if "weights" in options and "step" in options:
            assert isinstance(options["weights"], np.ndarray)
            assert isinstance(options["step"], int)
            self.current_weight = np.array2string(options["weights"], precision=2, separator=',')
            self.weight_id += 1
            self.current_step = options["step"]

        observations = super().reset(**kwargs)
        self.terminated = False
        self.truncated = False
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.recorded_frames = []
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        elif self.weight_trigger:
            video_name = f"{self.name_prefix}-step{self.current_step}-weight-{self.current_weight}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=base_path,
            metadata={
                "step_id": self.step_id, 
                "episode_id": self.episode_id,
                "num_evaluated_weights": self.weight_id,
                "evaluated_weight": self.current_weight
            },
            disable_logger=self.disable_logger,
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        elif self.episode_trigger:
            return self.episode_trigger(self.episode_id)
        elif self.weight_trigger:
            return self.weight_trigger(self.weight_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            # increment steps and episodes
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                self.video_recorder.capture_frame()
                self.recorded_frames += 1
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 1

    def render(self, *args, **kwargs):
        """Compute the render frames as specified by render_mode attribute during initialization of the environment or as specified in kwargs."""
        if self.video_recorder is None or not self.video_recorder.enabled:
            return super().render(*args, **kwargs)

        if len(self.video_recorder.render_history) > 0:
            recorded_frames = [
                self.video_recorder.render_history.pop()
                for _ in range(len(self.video_recorder.render_history))
            ]
            if self.recording:
                return recorded_frames
            else:
                return recorded_frames + super().render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()