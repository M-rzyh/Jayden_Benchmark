from abc import ABC, abstractmethod
from distutils.util import strtobool
from typing import Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import mo_gymnasium as mo_gym
import wandb

from pymoo.util.ref_dirs import get_reference_directions
from mo_utils.pareto import filter_pareto_dominated
from mo_utils.performance_indicators import (
    cardinality,
    expected_utility,
    hypervolume,
    sparsity,
)
from mo_utils.weights import equally_spaced_weights

# TODO: implement this for all dr envs
class DREnv(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset_random(self):
        pass

    @abstractmethod
    def reset_agent(self):
        pass

    @property
    def encoding(self):
        pass


def make_env(gym_id, algo_name, record_video, record_video_freq, **kwargs):
    env = gym.make(gym_id, 
                   render_mode="rgb_array" if record_video else None, 
                   **kwargs)
    if record_video:
        env = RecordVideo(
            env, 
            f"videos/{algo_name}/{gym_id}/", 
            episode_trigger=lambda t: t % record_video_freq == 0,
            disable_logger=True
        )
    return env

class RandomMOEnvWrapper(gym.Wrapper):
    def __init__(self, 
                 env: gym.Wrapper,
                 algo_name: str,
                 seed: int,
                 generalization_algo: str,
                 test_envs: List[str],
                 record_video: bool,
                 record_video_freq: int = 200, # len(eval_weights) * rep % record_video_freq == 0
                 save_metrics: List[str] = ['hv', 'eum'],
                 **kwargs):
        super().__init__(env)
        self.is_dr = generalization_algo == 'domain_randomization'
        self.algo_name = algo_name
        self.test_env_names = test_envs
        make_fn = [
            lambda env_name=env_name: make_env(env_name, algo_name, record_video, record_video_freq, **kwargs) for env_name in test_envs
        ]
        self.test_envs = mo_gym.MOSyncVectorEnv(make_fn)

        self.save_metrics = save_metrics
        self.best_metrics = [[-np.inf for _ in range(len(test_envs))] for _ in save_metrics]
        self.seed = seed
        

    # def step(self, action):
    #     ob, reward, terminated, truncated, info = super().step(action)

    #     if terminated:
    #         if self.is_dr:
    #             self.reset_random()
    #             # ob = self.reset_agent()
    #         else:
    #             ob = self._reset()

    #     return ob, reward, terminated, truncated, info
    
    # def reset(self, *, seed=None, options=None):
    #     if self.is_dr:
    #         print('Environment reset!')
    #         return self.env.unwrapped.reset_random()
        
    #     return super().reset()
    

    def eval_mo(
        self,
        agent,
        w: Optional[np.ndarray] = None,
        scalarization=np.dot,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluates one episode of the agent in the vectorised test environments.

        Args:
            agent: Agent
            scalarization: scalarization function, taking weights and reward as parameters
            w (np.ndarray): Weight vector

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return. 
            Each is an array where each element corresponds to a sub-environment in the vectorized environment.
        """
        obs, _ = self.test_envs.reset()
        done = np.array([False] * self.test_envs.num_envs)
        vec_return = np.zeros((self.test_envs.num_envs, len(w)))
        disc_vec_return = np.zeros_like(vec_return)
        gamma = np.ones(self.test_envs.num_envs)
        mask = np.ones(self.test_envs.num_envs, dtype=bool)

        while not all(done):
            actions = agent.eval(obs, np.tile(w, (self.test_envs.num_envs, 1)), num_envs = self.test_envs.num_envs)
            obs, r, terminated, truncated, info = self.test_envs.step(actions)
            mask &= ~terminated  # Update the mask
            vec_return[mask] += r[mask]
            disc_vec_return[mask] += gamma[mask, None] * r[mask]
            gamma[mask] *= agent.gamma
            done |= np.logical_or(terminated, truncated)

        if w is None:
            scalarized_return = scalarization(vec_return)
            scalarized_discounted_return = scalarization(disc_vec_return)
        else:
            w_tiled = np.tile(w, (self.test_envs.num_envs, 1))
            scalarized_return = np.einsum('ij,ij->i', vec_return, w_tiled)
            scalarized_discounted_return = np.einsum('ij,ij->i', disc_vec_return, w_tiled)

        return (scalarized_return, scalarized_discounted_return, vec_return, disc_vec_return)


    def policy_evaluation_mo(
        self, agent, w: Optional[np.ndarray], scalarization=np.dot, rep: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates the value of a policy by running the policy for multiple episodes. Returns the average returns.

        Args:
            agent: Agent
            env: MO-Gymnasium environment
            w (np.ndarray): Weight vector
            scalarization: scalarization function, taking reward and weight as parameters
            rep (int, optional): Number of episodes for averaging. Defaults to 5.

        Returns:
            (float, float, np.ndarray, np.ndarray): Avg scalarized return, Avg scalarized discounted return, Avg vectorized return, Avg vectorized discounted return
        """
        evals = [self.eval_mo(agent=agent, w=w, scalarization=scalarization) for _ in range(rep)]
        avg_scalarized_return = np.mean([eval[0] for eval in evals], axis=0)
        avg_scalarized_discounted_return = np.mean([eval[1] for eval in evals], axis=0)
        avg_vec_return = np.mean([eval[2] for eval in evals], axis=0)
        avg_disc_vec_return = np.mean([eval[3] for eval in evals], axis=0)

        return (
            avg_scalarized_return,
            avg_scalarized_discounted_return,
            avg_vec_return,
            avg_disc_vec_return,
        )
    
    def log_all_multi_policy_metrics(
        self,
        agent,
        current_fronts: np.ndarray,
        hv_ref_point: np.ndarray,
        reward_dim: int,
        global_step: int,
        n_sample_weights: int,
        discounted: bool,
    ):
        """Logs all metrics for multi-policy training (one for each test environment).

        Logged metrics:
        - hypervolume
        - sparsity
        - expected utility metric (EUM)

        Args:
            current_fronts: List of current Pareto front approximations, has shape of (num_test_envs, num_eval_weights, num_objectives)
            hv_ref_point: reference point for hypervolume computation
            reward_dim: number of objectives
            global_step: global step for logging
            n_sample_weights: number of weights to sample for EUM and MUL computation
            ref_front: reference front, if known
            discounted: if using discounted values or not
        """
        disc_str = "discounted_" if discounted else ""
        for i, current_front in enumerate(current_fronts):
            filtered_front = list(filter_pareto_dominated(current_front))
            hv = hypervolume(hv_ref_point, filtered_front)
            sp = sparsity(filtered_front)
            eum = expected_utility(filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights))
            card = cardinality(filtered_front)

            wandb.log(
                {
                    f"eval/{disc_str}hypervolume/{self.test_env_names[i]}": hv,
                    f"eval/{disc_str}sparsity/{self.test_env_names[i]}": sp,
                    f"eval/{disc_str}eum/{self.test_env_names[i]}": eum,
                    f"eval/{disc_str}cardinality/{self.test_env_names[i]}": card,
                    "global_step": global_step,
                },
                commit=False,
            )
            front = wandb.Table(
                columns=[f"objective_{j}" for j in range(1, reward_dim + 1)],
                data=[p.tolist() for p in filtered_front],
            )
            wandb.log({f"eval/{disc_str}front/{self.test_env_names[i]}": front})

            metrics = {
                'hv': hv,
                'sp': sp,
                'eum': eum,
                'card': card
            }

            for j, save_metric in enumerate(self.save_metrics):
                if save_metric in metrics.keys():
                    if metrics[save_metric] > self.best_metrics[j][i]:
                        self.best_metrics[j][i] = hv
                        agent.save(
                            save_dir=f"weights/{agent.experiment_name}/best_{save_metric}/seed{self.seed}",
                            filename=f"{self.test_env_names[i]}", 
                            save_replay_buffer=False
                        )


    def _report(
        self,
        scalarized_return: np.ndarray,
        scalarized_discounted_return: np.ndarray,
        vec_return: np.ndarray,
        disc_vec_return: np.ndarray,
        global_step: int
    ):
        """Logs the evaluation metrics.

        Args:
            scalarized_return: scalarized return
            scalarized_discounted_return: scalarized discounted return
            vec_return: vectorized return
            disc_vec_return: vectorized discounted return
        """
        for i, returns in enumerate(vec_return):
            metrics = {
                f"eval/scalarized_return/{self.test_env_names[i]}": scalarized_return[i],
                f"eval/scalarized_discounted_return/{self.test_env_names[i]}": scalarized_discounted_return[i],
                "global_step": global_step
            }
            for j in range(returns.shape[0]):
                metrics.update({
                    f"eval/vec_{j}/{self.test_env_names[i]}": vec_return[i][j],
                    f"eval/discounted_vec_{j}/{self.test_env_names[i]}": disc_vec_return[i][j]
                })
            wandb.log(metrics)


    def eval(self, agent, eval_weights, rep, ref_point, reward_dim, global_step):
        print('Evaluating agent on test environments at step: ', global_step)
        scalarized_returns = []
        scalarized_discounted_returns = []
        vec_returns = []
        disc_vec_returns = []

        for ew in eval_weights:
            scalarized_return, scalarized_discounted_return, vec_return, disc_vec_return = self.policy_evaluation_mo(agent, ew, rep=rep)
            scalarized_returns.append(scalarized_return)
            scalarized_discounted_returns.append(scalarized_discounted_return)
            vec_returns.append(vec_return)
            disc_vec_returns.append(disc_vec_return)

        scalarized_returns = np.stack(scalarized_returns, axis=1)
        scalarized_discounted_returns = np.stack(scalarized_discounted_returns, axis=1)
        mean_scalarized_returns = np.mean(scalarized_returns, axis=1)
        mean_scalarized_discounted_returns = np.mean(scalarized_discounted_returns, axis=1)
        mean_vec_returns = np.mean(vec_returns, axis=0)
        mean_disc_vec_returns = np.mean(disc_vec_returns, axis=0)

        self._report(
            mean_scalarized_returns,
            mean_scalarized_discounted_returns,
            mean_vec_returns,
            mean_disc_vec_returns,
            global_step=global_step
        )

        vec_returns = np.stack(vec_returns, axis=1)
        disc_vec_returns = np.stack(disc_vec_returns, axis=1)
        n_sample_weights = len(eval_weights)
        
        # Undiscounted values
        # self.log_all_multi_policy_metrics(
        #     current_fronts=vec_returns,
        #     hv_ref_point=ref_point,
        #     reward_dim=reward_dim,
        #     global_step=global_step,
        #     n_sample_weights=n_sample_weights,
        #     discounted=False
        # )
        for i, current_front in enumerate(vec_returns):
            filtered_front = list(filter_pareto_dominated(current_front))
            front = wandb.Table(
                columns=[f"objective_{j}" for j in range(1, reward_dim + 1)],
                data=[p.tolist() for p in filtered_front],
            )
            wandb.log({f"eval/front/{self.test_env_names[i]}": front})

        # Discounted values
        self.log_all_multi_policy_metrics(
            agent=agent,
            current_fronts=disc_vec_returns,
            hv_ref_point=ref_point,
            reward_dim=reward_dim,
            global_step=global_step,
            n_sample_weights=n_sample_weights,
            discounted=True
        )
