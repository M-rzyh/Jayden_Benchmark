import gymnasium as gym
from envs.mo_lava_gap.mo_lava_gap import MOLavaGapDR

class MOLavaGapPool(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        super().__init__(bit_map=bit_map, **kwargs)

class MOLavaGapCreek(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0]
        ]
        super().__init__(bit_map=bit_map, **kwargs)

class MOLavaGapMaze(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0]
        ]
        super().__init__(bit_map=bit_map, **kwargs)

class MOLavaGapSnake(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0]
        ]
        super().__init__(bit_map=bit_map, **kwargs)
                         
def register_lava_gap():
    try:
        gym.envs.register(
            id="MOLavaGapDR-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap:MOLavaGapDR",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapPool-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapPool",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapCreek-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapCreek",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapMaze-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapMaze",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapSnake-v0", # copy of the dr environment but renamed for clarity
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapSnake",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
