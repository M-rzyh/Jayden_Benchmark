from ued_mo_envs.mo_car_racing.old.car_racing_bezier import CarRacingBezier
from gym import spaces
import numpy as np

class MOCarRacing(CarRacingBezier):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		# Result reward, shaping reward, main engine cost, side engine cost
		self.reward_space = spaces.Box(
			low=np.array([-100, -np.inf, -1, -1]),
			high=np.array([100, np.inf, 0, 0]),
			shape=(4,),
			dtype=np.float32,
		)
		self.reward_dim = 4
	
	def step(self, action):
		