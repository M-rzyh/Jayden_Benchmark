import numpy as np

from mo_utils.evaluation import eval_mo
from algos.multi_policy.capql.capql import CAPQL

from ued_mo_envs.registration import make as gym_make
from ued_mo_envs.ued_env_wrapper import UEDMOEnvWrapper
from mo_utils.evaluation import seed_everything

import gymnasium as gym

def register_envs():
    try:
        gym.envs.register(
            id="MOLunarLanderUED-v0",
            entry_point="ued_mo_envs.mo_lunarlander.adversarial:MOLunarLanderUED",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalOne",
            entry_point="ued_mo_envs.mo_lunarlander.lunarlander_test_envs:LunarLanderEvalOne",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalTwo",
            entry_point="ued_mo_envs.mo_lunarlander.lunarlander_test_envs:LunarLanderEvalTwo",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalThree",
            entry_point="ued_mo_envs.mo_lunarlander.lunarlander_test_envs:LunarLanderEvalThree",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalFour",
            entry_point="ued_mo_envs.mo_lunarlander.lunarlander_test_envs:LunarLanderEvalFour",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

def test_capql_dr():
    register_envs()
    seed_everything(0)
    env = gym.make("MOLunarLanderUED-v0", continuous=True)
    eval_env = gym.make("MOLunarLanderUED-v0", continuous=True)
    env = UEDMOEnvWrapper(env, 
                          ued_algo="domain_randomization", 
                          test_env=[
                                "MOLunarLanderUED-v0",
                                "LunarLanderEvalOne",
                                "LunarLanderEvalTwo",
                                "LunarLanderEvalThree",
                                "LunarLanderEvalFour"
                              ],
                          continuous=True)

    agent = CAPQL(
        env,
        log=False,  
        is_ued=True
    )

    agent.train(
        total_timesteps=100000,
        eval_env=eval_env,
        ref_point=np.array([0.0, 0.0]),
        eval_freq=1,
    )

    scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = eval_mo(agent, env=eval_env, w=np.array([0.25, 0.25, 0.25, 0.25]))
    assert scalar_return != 0
    assert scalarized_disc_return != 0
    assert len(vec_ret) == 4
    assert len(vec_disc_ret) == 4
    print("scalar_return:", scalar_return, "scalarized_disc_return:", scalarized_disc_return, "vec_ret:", vec_ret, "vec_disc_ret:", vec_disc_ret)


if __name__ == "__main__":
    test_capql_dr()