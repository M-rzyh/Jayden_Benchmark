from typing import Optional

class DRRunner:
    """
	Orchestrates rollouts across vectorized environments, and include logic
    for domain randomization.
	"""
    def __init__(
        self,
        env_name: str,
        env_kwargs: dict,
        n_parallel: int = 1,
        n_eval_episodes: int = 1,
        n_rollout_steps: int = 256,
        lr: float = 1e-4,
        lr_final: Optional[float] = None,
        lr_anneal_steps: int = 0,
        max_grad_norm: float = 0.5,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        adam_eps: float = 1e-5,
        normalive_returns: bool = False,
        track_env_metrics: bool = False,
        render: bool = False,
        device: str = 'cpu'):

        self.env_name = env_name
        self.n_parallel = n_parallel
        self.
