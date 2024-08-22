import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import random

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv
import operator
from functools import reduce
from typing import Optional

from morl_generalization.algos.dr import DREnv

class MOLavaGapDR(MiniGridEnv, DREnv):
    """
    ## Description
    Multi-objective version of the Minigrid Lava Gap environment. 
    (https://minigrid.farama.org/environments/minigrid/LavaGapEnv/)
    Unlike the original Lava Gap environment, the episode does not terminate when the agent falls
    into lava.

    ## Parameters
    - size: maze is size*size big (including walls)
    - agent_start_pos: agent starting position
    - agent_start_dir: agent starting direction
    - goal_pos: goal default position. If None, default to bottom right corner.
    - n_lava: max number of lava tiles to add
    - bit_map: (size-1)*(size-1) list to indicate maze configuration (1 for lava, 0 for empty)

    ## Reward Space
    The reward is 2-dimensional:
    - 0: Lava Damage (-1 if agent falls into lava)
    - 1: Time Penalty (-1 for every step the agent has taken)

    ## Termination
    The episode ends if any one of the following conditions is met:
    - 1: The agent reaches the goal. +100 will be added to all dimensions in the vectorial reward.
    - 2: Timeout (see max_steps).
    """

    def __init__(
        self,
        size=11,
        max_steps=256,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_pos=None,
        n_lava=1000,
        bit_map=None,
        is_rgb=False,
        tile_size=8,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Reduce lava if there are too many
        self.n_lava = min(int(n_lava), (size-2)**2 - 2)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True, # Set this to True for maximum speed
            highlight=False, # Fully observable
            **kwargs,
        )

        self.goal_pos = (self.width - 2, self.height - 2) if goal_pos is None else goal_pos

        self._gen_grid(self.width, self.height)
        # Default configuration (if provided)
        if bit_map is not None:
            bit_map = np.array(bit_map)
            assert bit_map.shape == (size-2, size-2), "invalid bit map configuration"
            indices = np.argwhere(bit_map)
            for y, x in indices:
                # Add an offset of 1 for the outer walls
                self.put_obj(Lava(), x + 1, y + 1)

        # Observation space
        self.is_rgb = is_rgb
        if self.is_rgb:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.width * tile_size, self.height * tile_size, 3),
                dtype='uint8'
            )
        else:
            imgShape = (self.width, self.height, 3)
            imgSize = reduce(operator.mul, imgShape, 1)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(imgSize,),
                dtype='float32'
            )
        # lava damage, time penalty
        self.reward_space = spaces.Box(
            low=np.array([-max_steps, -max_steps]),
            high=np.array([100, 100]),
            shape=(2,),
            dtype=np.float32,
        )
        self.reward_dim = 2

        # Only 3 actions permitted: turn left, turn right, move forward
        self.action_space = spaces.Discrete(self.actions.forward + 1)

    @staticmethod
    def _gen_mission():
        return "get to goal while avoiding lava, or saving time, or both"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def step(self, action):
        # Invalid action
        if action >= self.action_space.n or action < 0:
            raise ValueError("Invalid action!")
        
        self.step_count += 1

        vec_reward = np.zeros(2, dtype=np.float32) 
        vec_reward[1] = -1 # -1 for time penalty
        terminated = False
        truncated = False
        
        # Get the contents of the current cell (penalize if the agent stays in lava cells)
        current_cell = self.grid.get(*self.agent_pos)
        in_lava = current_cell is not None and current_cell.type == "lava"

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            if in_lava: 
                # stay in lava
                vec_reward[0] = -5

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            if in_lava: 
                # stay in lava
                vec_reward[0] = -5

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                # +100 to all objectives if reach goal
                vec_reward += 100
            if fwd_cell is not None and fwd_cell.type == "lava": 
                # walk into lava
                vec_reward[0] = -5
        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        if self.is_rgb:
            return self.rgb_observation(), vec_reward, terminated, truncated, {}
        
        return self.observation(), vec_reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # Step count since episode start
        self.step_count = 0

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        if self.render_mode == "human":
            self.render()

        if self.is_rgb:
            return self.rgb_observation(), {}
        return self.observation(), {}
    
    def _resample_n_lava(self):
        n_lava = np.random.randint(0, self.n_lava)

        return n_lava
    
    def reset_random(self):
        """Use domain randomization to create the environment."""
        # Create an empty grid with goal and agent only
        self._gen_grid(self.width, self.height)

        # prevent lava from being place on start and goal positions
        def reject_fn(self, pos):
            if pos == self.agent_start_pos or pos == self.goal_pos:
                return True
            return False
        
        grid_width = self.width - 2
        grid_height = self.height - 2

        # Create a list of all possible positions
        all_positions = [(x, y) for x in range(grid_width) for y in range(grid_height)]

        # Remove the agent and goal positions
        all_positions.remove((0, 0))
        all_positions.remove((grid_width - 1, grid_height - 1))

        # Randomly sample `n_lava` unique positions
        n_lava = self._resample_n_lava()
        selected_positions = random.sample(all_positions, n_lava)

        # Place the Lava objects at the randomly selected positions
        for x, y in selected_positions:
            self.put_obj(Lava(), x + 1, y + 1)
        
        # Double check that the agent and goal doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_start_pos)
        assert start_cell is None or start_cell.can_overlap(), "agent's initial position is invalid"

    def observation(self):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        full_grid = full_grid.flatten()
        return full_grid/ 1.
    
    def rgb_observation(self):
        env = self.unwrapped
        rgb_img = self.grid.render(
            self.tile_size,
            env.agent_pos,
            env.agent_dir,
        )
        return rgb_img

if __name__ == "__main__":
    from gymnasium.envs.registration import register
    from mo_utils.evaluation import seed_everything
    import matplotlib.pyplot as plt

    seed_everything(42)

    register(
        id="MOLavaGapDR",
        entry_point="envs.mo_lava_gap.mo_lava_gap:MOLavaGapDR",
    )
    env = gym.make(
        "MOLavaGapDR", 
        render_mode="human",
        # is_rgb=True,
    )

    terminated = False
    env.unwrapped.reset_random()
    env.reset()
    while True:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        print(r, terminated, truncated, obs.shape)
        # plt.figure()
        # plt.imshow(obs, vmin=0, vmax=255)
        # plt.show()
        env.render()
        if terminated or truncated:
            env.unwrapped.reset_random()
            env.reset()