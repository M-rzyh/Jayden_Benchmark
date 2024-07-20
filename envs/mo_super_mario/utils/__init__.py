from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def wrap_mario(env):
    from gymnasium.wrappers import (
        FrameStack,
        GrayScaleObservation,
        ResizeObservation,
        TimeLimit,
    )
    from mo_gymnasium.envs.mario.joypad_space import JoypadSpace
    from mo_gymnasium.utils import MOMaxAndSkipObservation

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MOMaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, max_episode_steps=1000)
    return env