from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from envs.mo_super_mario.utils.mario_video_wrapper import RecordMarioVideo

def wrap_mario(env, record_video=False, gym_id="", algo_name="", seed=0, record_video_freq=0):
    from gymnasium.wrappers import (
        FrameStack,
        GrayScaleObservation,
        ResizeObservation,
        TimeLimit,
    )
    from mo_gymnasium.envs.mario.joypad_space import JoypadSpace
    from mo_gymnasium.utils import MOMaxAndSkipObservation

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = TimeLimit(env, max_episode_steps=1000) # this must come before video recording else truncation will not be captured
    if record_video:
        env = RecordMarioVideo(
            env, 
            f"videos/{algo_name}/seed{seed}/{gym_id}/", 
            episode_trigger=lambda t: t % record_video_freq == 0,
            disable_logger=True
        )
    env = MOMaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    return env