import gymnasium as gym

def register_envs():
    # ================== Registering LunarLander ==================
    try:
        gym.envs.register(
            id="MOLunarLanderUED-v0",
            entry_point="ued_mo_envs.mo_lunar_lander.adversarial:MOLunarLanderUED",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalOne",
            entry_point="ued_mo_envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalOne",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalTwo",
            entry_point="ued_mo_envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalTwo",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalThree",
            entry_point="ued_mo_envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalThree",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalFour",
            entry_point="ued_mo_envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalFour",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    # ================== Registering BipedalWalker ==================
    try:
        gym.envs.register(
            id='MOBipedalWalkerUED-v0',
            entry_point="ued_mo_envs.mo_bipedal_walker.bipedal_walker_randomized:MOBipedalWalkerUED",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id='BipedalWalker-Med-Stairs-v0',
            entry_point="ued_mo_envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedStairs",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id='BipedalWalker-Med-PitGap-v0',
            entry_point="ued_mo_envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedPits",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="BipedalWalker-Med-StumpHeight-v0",
            entry_point="ued_mo_envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedStumps",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


    try:
        gym.envs.register(
            id="BipedalWalker-Med-Roughness-v0",
            entry_point="ued_mo_envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedRoughness",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")