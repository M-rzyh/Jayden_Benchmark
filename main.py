import numpy as np

from mo_utils.evaluation import policy_evaluation_mo
from algos.multi_policy.capql.capql import CAPQL

from envs.generalization_evaluator import MORLGeneralizationEvaluator
from envs.register_envs import register_envs
from mo_utils.evaluation import seed_everything

import gymnasium as gym

def test_capql_dr():
    register_envs()
    seed_everything(0)
    # env = gym.make("MOLunarLanderDR-v0", continuous=True)
    # eval_env = gym.make("MOLunarLanderDR-v0", continuous=True)
    # env = MORLGeneralizationEvaluator(env, 
    #                       generalization_algo="domain_randomization", 
    #                       test_env=[
    #                             "MOLunarLanderDR-v0",
    #                             "LunarLanderEvalOne",
    #                             "LunarLanderEvalTwo",
    #                             "LunarLanderEvalThree",
    #                             "LunarLanderEvalFour"
    #                           ],
    #                       continuous=True)

    # env = gym.make("MOBipedalWalkerDR-v0")
    # eval_env = gym.make("MOBipedalWalkerDR-v0")
    # env = MORLGeneralizationEvaluator(env, 
    #                       generalization_algo="domain_randomization", 
    #                       test_env=[
    #                             "BipedalWalker-v3",
    #                             "BipedalWalkerHardcore-v3",
    #                             "BipedalWalker-Med-Stairs-v0",
    #                             "BipedalWalker-Med-PitGap-v0",
    #                             "BipedalWalker-Med-StumpHeight-v0",
    #                             "BipedalWalker-Med-Roughness-v0"
    #                         ])
    # env = gym.make("MOHopperDR-v5")
    # eval_env = gym.make("MOHopperDR-v5")
    # env = MORLGeneralizationEvaluator(env, 
    #                       generalization_algo="domain_randomization", 
    #                       test_env=[
    #                             "MOHopperDR-v5",
    #                             "MOHopperLight-v5",
    #                             "MOHopperHeavy-v5",
    #                             "MOHopperSlippery-v5",
    #                             "MOHopperHighDamping-v5",
    #                       ])

    env = gym.make("MOHalfCheetahDR-v5")
    eval_env = gym.make("MOHalfCheetahDR-v5")
    env = MORLGeneralizationEvaluator(env, 
                          generalization_algo="domain_randomization", 
                          test_envs=[
                                "MOHalfCheetahLight-v5",
                                "MOHalfCheetahHeavy-v5",
                                "MOHalfCheetahSlippery-v5",
                                "MOHalfCheetahHard-v5",
                                "MOHalfCheetahDR-v5"
                          ])
    
    agent = CAPQL(
        env,
        log=False,  
    )

    agent.train(
        total_timesteps=3000000,
        eval_env=eval_env,
        ref_point=np.array([0.0, 0.0]),
        eval_freq=1,
        test_generalization=True
    )

    weights = [
        np.array([0.8, 0.2]),
        np.array([0.7, 0.3]),
        np.array([0.6, 0.4]),
        np.array([0.5, 0.5]),
        np.array([0.4, 0.6]),
        np.array([0.3, 0.7]),
        np.array([0.2, 0.8]),
        np.array([0.1, 0.9]),
    ]

    # results = []

    with open('results.txt', 'w') as f:
        for w in weights:
            scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = policy_evaluation_mo(agent, env=eval_env, w=w, rep=10)
            f.write(f"Weights: {w}, Scalar Return: {scalar_return}, Scalarized Discounted Return: {scalarized_disc_return}, Vector Return: {vec_ret}, Vector Discounted Return: {vec_disc_ret}\n")


    # scalar_return, scalarized_disc_return, vec_ret, vec_disc_ret = policy_evaluation_mo(agent, env=eval_env, w=np.array([0.8, 0.2]), rep=10)
    # assert scalar_return != 0
    # assert scalarized_disc_return != 0
    # assert len(vec_ret) == 3
    # assert len(vec_disc_ret) == 3
    # print("results:", results)


if __name__ == "__main__":
    test_capql_dr()