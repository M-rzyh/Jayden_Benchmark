from envs.mo_mujoco.mo_hopper_randomized import MOHopperDR
from envs.mo_mujoco.mo_halfcheetah_randomized import MOHalfCheehtahDR
import numpy as np

# ============================ Hopper ============================
class MOHopperLight(MOHopperDR):
    def __init__(self):
        masses = np.array([0.5, 0.5, 0.3, 0.7])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperHeavy(MOHopperDR):
    def __init__(self):
        masses = np.array([9.0, 9.0, 8.5, 10.0])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperSlippery(MOHopperDR):
    def __init__(self):
        masses = np.array([3.7, 4.0, 2.8, 5.3])
        damping = np.array([1.0, 1.0, 1.0])
        friction = np.array([0.1])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperLowDamping(MOHopperDR):
    def __init__(self):
        masses = np.array([3.7, 4.0, 2.8, 5.3])
        damping = np.array([0.1, 0.1, 0.1])
        friction = np.array([1.0])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

class MOHopperHard(MOHopperDR):
    def __init__(self):
        masses = np.array([0.1, 9.0, 9.0, 0.1])
        damping = np.array([0.1, 0.1, 0.1])
        friction = np.array([0.1])
        task = np.concatenate([masses, damping, friction])
        super().__init__(task=task)

# ============================ Cheetah ============================
class MOHalfCheehtahLight(MOHalfCheehtahDR):
    def __init__(self):
        masses = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        friction = np.array([0.4])
        task = np.concatenate([masses, friction])
        super().__init__(task=task)

class MOHalfCheehtahHeavy(MOHalfCheehtahDR):
    def __init__(self):
        masses = np.array([10.0, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5])
        friction = np.array([0.4])
        task = np.concatenate([masses, friction])
        super().__init__(task=task)

class MOHalfCheehtahSlippery(MOHalfCheehtahDR):
    def __init__(self):
        masses = np.array([6.25020921, 1.54351464, 1.5874477, 1.09539749, 1.43807531, 1.20083682, 0.88451883])
        friction = np.array([0.02])
        task = np.concatenate([masses, friction])
        super().__init__(task=task)

class MOHalfCheehtahHard(MOHalfCheehtahDR):
    def __init__(self):
        masses = np.array([10.0, 0.1, 10.0, 10.0, 0.1, 0.1, 10.0])
        friction = np.array([0.02])
        task = np.concatenate([masses, friction])
        super().__init__(task=task)