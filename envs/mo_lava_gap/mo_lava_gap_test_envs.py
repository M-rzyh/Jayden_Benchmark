import gymnasium as gym
from envs.mo_lava_gap.mo_lava_gap import MOLavaGapDR

class MOLavaGapCorridor(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        goal_pos = [(6, 7), (4, 5), (8, 5)]
        weightages = [0.6, 0.1, 0.3]
        agent_pos = (2, 6) # middle row left column
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, agent_start_pos=agent_pos, weightages=weightages, **kwargs)

class MOLavaGapIslands(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 0, 0]
        ]
        goal_pos = [(10, 2), (10, 10), (2, 10)]
        weightages = [0.02, 0.75, 0.23]
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, **kwargs)

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
        goal_pos = [(10, 2), (2, 10), (10, 10)]
        weightages = [0.05, 0.05, 0.9]
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, **kwargs)

class MOLavaGapSnake(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0]
        ]
        goal_pos = [(7, 10), (8, 2), (10, 10)]
        weightages = [0.2, 0.3, 0.5]
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, **kwargs)

class MOLavaGapRoom(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0]
        ]
        goal_pos = [(9, 9), (3, 3), (9, 3)]
        weightages = [0.5, 0.3, 0.2]
        agent_pos = (6, 6) # center of the grid
        agent_dir = 3 # face upwards
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_dir=agent_dir, agent_start_pos=agent_pos, **kwargs)

class MOLavaGapLabyrinth(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0]
        ]
        goal_pos = [(6, 6), (8, 8), (10, 10)]
        weightages = [0.5, 0.05, 0.45]
        agent_pos = (2, 10) # bottom left
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_pos=agent_pos, **kwargs)

class MOLavaGapSmiley(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        goal_pos = [(5, 3), (8, 3), (6, 7)]
        weightages = [0.4, 0.4, 0.2]
        agent_pos = (2, 10) # bottom left
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_pos=agent_pos, **kwargs)

class MOLavaGapCheckerBoard(MOLavaGapDR):
    def __init__(self, **kwargs):
        bit_map = [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0]
        ]
        goal_pos = [(9, 3), (9, 9), (3, 9)]
        weightages = [0.3, 0.1, 0.6]
        agent_pos = (6, 4) # bottom left
        agent_dir = 1 # face downwards
        super().__init__(bit_map=bit_map, goal_pos=goal_pos, weightages=weightages, agent_start_pos=agent_pos, agent_start_dir=agent_dir, **kwargs)

                         
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
            id="MOLavaGapCorridor-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapCorridor",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapIslands-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapIslands",
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
            id="MOLavaGapSnake-v0", 
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapSnake",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapRoom-v0", 
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapRoom",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapLabyrinth-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapLabyrinth",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapSmiley-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapSmiley",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOLavaGapCheckerBoard-v0",
            entry_point="envs.mo_lava_gap.mo_lava_gap_test_envs:MOLavaGapCheckerBoard",
            max_episode_steps=256,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
