import gymnasium as gym

def register_envs():
    # ================== Registering LunarLander ==================
    try:
        gym.envs.register(
            id="MOLunarLanderDR-v0",
            entry_point="envs.mo_lunar_lander.mo_lunar_lander_randomized:MOLunarLanderDR",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalOne",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalOne",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalTwo",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalTwo",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalThree",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalThree",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="LunarLanderEvalFour",
            entry_point="envs.mo_lunar_lander.lunarlander_test_envs:LunarLanderEvalFour",
            max_episode_steps=500,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    # ================== Registering BipedalWalker ==================
    try:
        gym.envs.register(
            id='MOBipedalWalkerDR-v0',
            entry_point="envs.mo_bipedal_walker.bipedal_walker_randomized:MOBipedalWalkerDR",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id='BipedalWalker-Med-Stairs-v0',
            entry_point="envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedStairs",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id='BipedalWalker-Med-PitGap-v0',
            entry_point="envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedPits",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="BipedalWalker-Med-StumpHeight-v0",
            entry_point="envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedStumps",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


    try:
        gym.envs.register(
            id="BipedalWalker-Med-Roughness-v0",
            entry_point="envs.mo_bipedal_walker.bipedalwalker_test_envs:BipedalWalkerMedRoughness",
            max_episode_steps=2000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


    # ================== Registering Mujoco ==================

    # HalfCheetah
    try:
        gym.envs.register(
            id="MOHalfCheehtahDR-v5",
            entry_point="envs.mo_mujoco.mo_halfcheetah_randomized:MOHalfCheehtahDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHalfCheehtahLight-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheehtahLight",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHalfCheehtahHeavy-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheehtahHeavy",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    
    try:
        gym.envs.register(
            id="MOHalfCheehtahSlippery-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheehtahSlippery",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")


    try:
        gym.envs.register(
            id="MOHalfCheehtahHard-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHalfCheehtahHard",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    # Hopper
    try:
        gym.envs.register(
            id="MOHopperDR-v5",
            entry_point="envs.mo_mujoco.mo_hopper_randomized:MOHopperDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperLight-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperLight",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperHeavy-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperHeavy",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperSlippery-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperSlippery",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    try:
        gym.envs.register(
            id="MOHopperLowDamping-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperLowDamping",
            max_episode_steps=1000,
        )

    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    
    try:
        gym.envs.register(
            id="MOHopperHard-v5",
            entry_point="envs.mo_mujoco.mo_mujoco_test_envs:MOHopperHard",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")

    
    # Humanoid
    try:
        gym.envs.register(
            id="MOHumanoidDR-v0",
            entry_point="envs.mo_mujoco.mo_humanoid_randomized:MOHumanoidDR",
            max_episode_steps=1000,
        )
    except Exception as e:
        print(f"Unexpected error: {e}, {type(e)}")
