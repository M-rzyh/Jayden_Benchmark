from envs.mo_lunar_lander.utils.mo_lunar_lander import MOLunarLander
import gymnasium as gym

#fixed params that were randomized initially
params = [[ -3.58211243,   0.20845414],
          [-10.02311556,   5.33210306],
          [-11.7806391 ,   6.08057095],
          [ -0.67134782,   8.74679624],
          [ -9.98520401,  15.78928294],
          [ -6.74740918,  17.31452842],
          [-11.06398877,  11.04122087],
          [ -9.34399539,  14.07790794],
          [ -2.36235808,   4.30473596],
          [ -4.46353633,  14.59934468]]

class LunarLanderEvalOne(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[0][0]
        wind_power = params[0][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalTwo(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[1][0]
        wind_power = params[1][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalThree(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[2][0]
        wind_power = params[2][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalFour(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[3][0]
        wind_power = params[3][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalFive(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[4][0]
        wind_power = params[4][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalSix(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[5][0]
        wind_power = params[5][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalSeven(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[6][0]
        wind_power = params[6][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalEight(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[7][0]
        wind_power = params[7][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalNine(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[8][0]
        wind_power = params[8][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)

class LunarLanderEvalTen(MOLunarLander):
    def __init__(self, continuous = True):
        gravity = params[9][0]
        wind_power = params[9][1]
        super().__init__(gravity = gravity, wind_power = wind_power, \
                        enable_wind = True, continuous = continuous)
        
def register_lunar_lander():
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
