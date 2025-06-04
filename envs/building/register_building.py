from gymnasium.envs.registration import register
from envs.building.env_building import BuildingEnv_3d, BuildingEnv_9d

def register_building():
    register(
        id="building-3d-v0",
        entry_point="envs.building.env_building:BuildingEnv_3d",
    )

    register(
        id="building-9d-v0",
        entry_point="envs.building.env_building:BuildingEnv_9d",
    )