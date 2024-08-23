import heapq
import json
import gymnasium as gym
import multiprocessing as mp

from envs.mo_lava_gap.mo_lava_gap_test_envs import register_lava_gap

def scalarize_reward(vec_reward, weights):
    return weights[0] * vec_reward[0] + weights[1] * vec_reward[1]

def a_star_search(env, weights):
    start_state = (env.unwrapped.agent_start_pos, env.unwrapped.agent_start_dir)
    goal_state = env.unwrapped.goal_pos

    frontier = [(0, start_state)]
    came_from = {}
    cost_so_far = {}
    came_from[start_state] = None
    cost_so_far[start_state] = 0

    while frontier:
        current_priority, current = heapq.heappop(frontier)

        if current[0] == goal_state:
            break

        for action in range(env.action_space.n):
            env.reset()
            env.unwrapped.agent_pos, env.unwrapped.agent_dir = current
            _, vec_reward, _, _, _ = env.step(action)
            scalar_reward = scalarize_reward(vec_reward, weights)

            new_cost = cost_so_far[current] + scalar_reward
            next_state = (env.unwrapped.agent_pos, env.unwrapped.agent_dir)

            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = (current, action)

    # Reconstruct path
    trajectory = []
    current = (env.unwrapped.agent_pos, env.unwrapped.agent_dir)
    while current != start_state:
        previous, action = came_from[current]
        trajectory.append(action)
        current = previous
    trajectory.reverse()
    return trajectory

def find_trajectory_for_env(env_name, weights):
    register_lava_gap()
    env = gym.make(env_name)
    trajectory = a_star_search(env, weights)
    return env_name, weights, trajectory

if __name__ == '__main__':
    env_names = ['MOLavaGapCreek-v0', 'MOLavaGapMaze-v0', 'MOLavaGapSnake-v0']
    weights_list = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]

    # Prepare arguments for multiprocessing
    tasks = [(env_name, weights) for env_name in env_names for weights in weights_list]

    # Use multiprocessing to parallelize the A* search
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(find_trajectory_for_env, tasks)

    # Aggregate results
    all_best_trajectories = {}
    for env_name, weights, trajectory in results:
        if env_name not in all_best_trajectories:
            all_best_trajectories[env_name] = {}
        all_best_trajectories[env_name][str(weights)] = trajectory

    # Save the results as JSON
    with open('best_trajectories.json', 'w') as f:
        json.dump(all_best_trajectories, f, indent=4)

    print("Done")
