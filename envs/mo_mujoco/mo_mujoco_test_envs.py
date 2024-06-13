from envs.mo_mujoco.mo_hopper_randomized import MOHopperUED
import numpy as np

class MOHopperLight(MOHopperUED):
    def __init__(self):
        masses = np.array([0.5, 0.5, 0.3, 0.7])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperHeavy(MOHopperUED):
    def __init__(self):
        masses = np.array([9.0, 9.0, 8.5, 10.0])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperSlippery(MOHopperUED):
    def __init__(self):
        masses = np.array([3.7, 4.0, 2.8, 5.3])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([0.1])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperHighDamping(MOHopperUED):
    def __init__(self):
        masses = np.array([3.7, 4.0, 2.8, 5.3])
        damping = np.array([3.0, 3.0, 3.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)