import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.record_video import RecordVideo

from envs.mo_super_mario.utils import wrap_mario
from morl_generalization.algos.dr import DRWrapper, DynamicsInObs, AsymmetricDRWrapper
from morl_generalization.wrappers import MORecordVideo, ActionHistoryWrapper, StateHistoryWrapper

# TODO: allow customisable history len. Currently using fixed history len of 3
def get_env_selection_algo_wrapper(env, env_selection_algo, history_len = 3, is_eval_env = False) -> gym.Env:
    if env_selection_algo == "domain_randomization": # randomizes domain every `reset` call
        return DRWrapper(env)
    elif env_selection_algo == "asymmetric_dr": # randomizes domain + asymmetric actor-critic
        if is_eval_env:
            return DRWrapper(env) # eval env should not provide any context
        return AsymmetricDRWrapper(env)
    elif env_selection_algo == "asymmetric_dr_state_history": # randomizes domain + asymmetric actor-critic + state history
        if is_eval_env:
            return StateHistoryWrapper(env, history_len) # eval env should not provide any context
        return AsymmetricDRWrapper(env, history_len, state_history=True)
    elif env_selection_algo == "asymmetric_dr_action_history": # randomizes domain + asymmetric actor-critic + action history
        if is_eval_env:
            return ActionHistoryWrapper(env, history_len) # eval env should not provide any context
        return AsymmetricDRWrapper(env, history_len, action_history=True)
    elif env_selection_algo == "asymmetric_dr_state_action_history": # randomizes domain + asymmetric actor-critic + state history + action history
        if is_eval_env:
            return ActionHistoryWrapper(StateHistoryWrapper(env, history_len), history_len) # eval env should not provide any context
        return AsymmetricDRWrapper(env, history_len, state_history=True, action_history=True)
    else:
        raise NotImplementedError

def make_test_envs(gym_id, algo_name, seed, generalization_algo, record_video=False, record_video_w_freq=None, record_video_ep_freq=None, **kwargs):
    is_mario = "mario" in gym_id.lower()
    if record_video:
        assert sum(x is not None for x in [record_video_w_freq, record_video_ep_freq]) == 1, "Must specify exactly one video recording trigger"
        if record_video_w_freq:
            print("Recording video every", record_video_w_freq, "weights evaluated")
        elif record_video_ep_freq:
            print("Recording video every", record_video_ep_freq, "episodes")

    if is_mario:
        env = gym.make(
                gym_id, 
                render_mode="rgb_array" if record_video else None, 
                death_as_penalty=True,
                time_as_penalty=True,
                **kwargs
            )
        env = wrap_mario(env, gym_id, algo_name, seed, record_video=record_video, record_video_ep_freq=record_video_ep_freq, record_video_w_freq=record_video_w_freq)
    else:
        env = gym.make(
                gym_id, 
                render_mode="rgb_array" if record_video else None, 
                **kwargs
            )
    
    if "highway" in gym_id.lower():
        env = FlattenObservation(env)
    
    # TODO: allow customisable history len. Currently using fixed history len of 3
    if generalization_algo == "asymmetric_dr_state_history":
        env =  StateHistoryWrapper(env)
    elif generalization_algo == "asymmetric_dr_action_history":
        env =  ActionHistoryWrapper(env)
    elif generalization_algo == "asymmetric_dr_state_action_history":
        env =  ActionHistoryWrapper(StateHistoryWrapper(env))

    if record_video and not is_mario:
        if record_video_w_freq: # record video every set number of weights evaluated
            env = MORecordVideo(
                env, 
                f"videos/{algo_name}/seed{seed}/{gym_id}/", 
                weight_trigger=lambda t: t % record_video_w_freq == 0,
                disable_logger=True
            )
        elif record_video_ep_freq: # record video every set number of episodes
            env = MORecordVideo(
                env, 
                f"videos/{algo_name}/seed{seed}/{gym_id}/", 
                episode_trigger=lambda t: t % record_video_ep_freq == 0,
                disable_logger=True
            )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
