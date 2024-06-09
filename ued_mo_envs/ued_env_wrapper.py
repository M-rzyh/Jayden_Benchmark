from abc import ABC, abstractmethod
from distutils.util import strtobool
from typing import Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from ued_mo_envs.registration import make as gym_make
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

class UEDEnv(ABC):
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


class UEDMOEnvWrapper(gym.Wrapper):
    def __init__(self, 
                 env: gym.Wrapper,
                 ued_algo: str,
                 test_env: List[str],
                 record_video: bool = False,
                 **kwargs):
        super().__init__(env)
        print(gym.envs.registry.keys())
        self.is_dr = ued_algo == 'domain_randomization'
        self.test_env_names = test_env
        make_fn = [
            lambda env_name=env_name: gym.make(env_name, **kwargs) for env_name in test_env
        ]
        self.test_env = gym.vector.SyncVectorEnv(make_fn)

    @staticmethod
    def make_env(env_name, record_video=False, **kwargs):
        env = gym.make(env_name, **kwargs)

        if record_video:
            from gym.wrappers.monitor import Monitor
            env = Monitor(env, "videos/", force=True)
            print('Recording video!', flush=True)
            
        return env

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
            render (bool, optional): Whether to render the environment. Defaults to False.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray): Scalarized return, scalarized discounted return, vectorized return, vectorized discounted return. 
            Each is an array where each element corresponds to a sub-environment in the vectorized environment.
        """
        obs = self.test_env.reset()
        done = np.array([False] * self.test_env.num_envs)
        vec_return = np.zeros((self.test_env.num_envs, len(w)))
        disc_vec_return = np.zeros_like(vec_return)
        gamma = np.ones(self.test_env.num_envs)
        mask = np.ones(self.test_env.num_envs, dtype=bool)

        while not all(done):
            action = agent.eval(obs, w)
            obs, r, terminated, truncated, info = self.test_env.step(action)
            mask &= ~terminated  # Update the mask
            vec_return[mask] += r[mask]
            disc_vec_return[mask] += gamma[mask, None] * r[mask]
            gamma[mask] *= agent.gamma
            done |= terminated

        if w is None:
            scalarized_return = scalarization(vec_return)
            scalarized_discounted_return = scalarization(disc_vec_return)
        else:
            scalarized_return = scalarization(w, vec_return)
            scalarized_discounted_return = scalarization(w, disc_vec_return)

        return scalarized_return, scalarized_discounted_return, vec_return, disc_vec_return


    def policy_evaluation_mo(
            self, agent, env, w: Optional[np.ndarray], scalarization=np.dot, rep: int = 5
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
            current_fronts: np.ndarray,
            hv_ref_point: np.ndarray,
            reward_dim: int,
            global_step: int,
            n_sample_weights: int,
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
            """
            for i, current_front in enumerate(current_fronts):
                filtered_front = list(filter_pareto_dominated(current_front))
                hv = hypervolume(hv_ref_point, filtered_front)
                sp = sparsity(filtered_front)
                eum = expected_utility(filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights))
                card = cardinality(filtered_front)

                wandb.log(
                    {
                        f"eval/hypervolume/{self.test_env_names[i]}": hv,
                        f"eval/sparsity/{self.test_env_names[i]}": sp,
                        f"eval/eum/{self.test_env_names[i]}": eum,
                        f"eval/cardinality/{self.test_env_names[i]}": card,
                        "global_step": global_step,
                    },
                    commit=False,
                )
                front = wandb.Table(
                    columns=[f"objective_{j}" for j in range(1, reward_dim + 1)],
                    data=[p.tolist() for p in filtered_front],
                )
                wandb.log({f"eval/front/{self.test_env_names[i]}": front})


    def eval(self, agent, eval_weights, rep, ref_point, reward_dim, global_step):
        print('Evaluating agent on test environments at step: ', global_step)
        returns_test_tasks = np.stack([
            self.policy_evaluation_mo(agent, ew, rep=rep)[3] for ew in eval_weights
        ], axis=1)
        n_sample_weights = len(eval_weights)
        self.log_all_multi_policy_metrics(
            current_front=returns_test_tasks,
            hv_ref_point=ref_point,
            reward_dim=reward_dim,
            global_step=global_step,
            n_sample_weights=n_sample_weights,
        )
