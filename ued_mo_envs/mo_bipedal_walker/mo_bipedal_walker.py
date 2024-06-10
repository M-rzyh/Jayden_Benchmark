import math

import numpy as np
from typing import List, Optional
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import (
        revoluteJointDef,
        polygonShape,
        edgeShape,
        fixtureDef
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

from gymnasium.envs.box2d.bipedal_walker import (
    SCALE,
    FPS,
    MOTORS_TORQUE,
    SPEED_HIP,
    SPEED_KNEE,
    LIDAR_RANGE,
    INITIAL_RANDOM,
    LEG_DOWN,
    LEG_H,
    VIEWPORT_W,
    VIEWPORT_H,
    TERRAIN_STEP,
    TERRAIN_LENGTH,
    TERRAIN_HEIGHT,
    TERRAIN_GRASS,
    TERRAIN_STARTPAD,
    HULL_FD,
    LEG_FD,
    LOWER_FD,
    FRICTION,
    ContactDetector,
    BipedalWalker
)
STAIR_HEIGHT_EPS = 1e-2


class MOBipedalWalker(BipedalWalker):  # no need for EzPickle, it's already in BipdalWalker
    """
    ## Description
    Multi-objective version of the BipedalWalker environment with customizable terrain.

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Moving forward reward
    - 1: Shaping reward (keeping head straight)
    - 2: Energy Cost

    ### Credits
    References:
    - [Gymnasium's env](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)
    - [Meta Research's DCD repo](https://github.com/facebookresearch/dcd)
    
    Adapted by Jayden Teoh, 2024 (https://github.com/JaydenTeoh)
    """

    def __init__(self, env_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_env_config(env_config) # set the environment configuration
        self._set_terrain_number() # set the terrain numbers

        # Result forward reward, balance reward, energy cost 
        # reward range is estimated for forward reward but not for balance reward and energy cost
        self.reward_space = spaces.Box(
            low=np.array([-100, 0, -1]),
            high=np.array([100, 1, 0]),
            shape=(3,),
            dtype=np.float32,
        )
        self.reward_dim = 3
        self.original_prev_shaping = None

    def re_init(self, env_config):
        self.set_env_config(env_config)

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0),
                                         (1, 0),
                                         (1, -1),
                                         (0, -1)]),
            friction=FRICTION)

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0),
                                      (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

    def set_env_config(self, env_config):
        self.config = env_config

    def _set_terrain_number(self):
        self.hardcore = False
        self.GRASS = 0
        self.STUMP, self.STAIRS, self.PIT = -1, -1, -1
        self._STATES_ = 1

        if self.config.stump_width and self.config.stump_height and self.config.stump_float:
            # STUMP exist
            self.STUMP = self._STATES_
            self._STATES_ += 1

        if self.config.stair_height and self.config.stair_width and self.config.stair_steps:
            # STAIRS exist
            self.STAIRS = self._STATES_
            self._STATES_ += 1

        if self.config.pit_gap:
            # PIT exist
            self.PIT = self._STATES_
            self._STATES_ += 1

        if self._STATES_ > 1:
            self.hardcore = True

    def _get_poly_stump(self, x, y, terrain_step):
        stump_width = self.np_random.integers(*self.config.stump_width)
        stump_height = self.np_random.uniform(*self.config.stump_height)
        stump_float = self.np_random.integers(*self.config.stump_float)

        countery = stump_height
        poly = [(x, y + stump_float * terrain_step),
                (x + stump_width * terrain_step, y + stump_float * terrain_step),
                (x + stump_width * terrain_step, y + countery * terrain_step + stump_float * terrain_step),
                (x, y + countery * terrain_step + stump_float * terrain_step), ]
        return poly

    def _generate_terrain(self, hardcore):
        # GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = self.GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        pit_diff = 0

        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == self.GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += self.config.ground_roughness * velocity

            elif state == self.PIT and oneshot:
                pit_gap = 1.0 + self.np_random.uniform(*self.config.pit_gap)
                counter = np.ceil(pit_gap)
                pit_diff = counter - pit_gap

                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == self.PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP
                if counter == 1:
                    self.terrain_x[-1] = self.terrain_x[-1] - pit_diff * TERRAIN_STEP
                    pit_diff = 0

            elif state == self.STUMP and oneshot:
                attempts = 0
                done = False
                while not done:
                    try:
                        poly = self._get_poly_stump(x, y, TERRAIN_STEP)
                        self.fd_polygon.shape.vertices = poly
                        done = True
                    except:
                        attempts += 1
                        if attempts > 10:
                            print("Stump issues: num attempts: ", attempts)
                            done = True

                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == self.STAIRS and oneshot:
                stair_height = self.np_random.uniform(
                    *self.config.stair_height)
                stair_slope = 1 if self.np_random.random() > 0.5 else -1
                stair_width = self.np_random.integers(*self.config.stair_width)
                stair_steps = self.np_random.integers(*self.config.stair_steps)
                original_y = y

                if stair_height > STAIR_HEIGHT_EPS:
                    for s in range(stair_steps):
                        poly = [(x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height * stair_slope) * TERRAIN_STEP),
                                (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height * stair_slope) * TERRAIN_STEP),
                                (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-stair_height + s * stair_height * stair_slope) * TERRAIN_STEP),
                                (x + (s * stair_width) * TERRAIN_STEP, y + (-stair_height + s * stair_height * stair_slope) * TERRAIN_STEP), ]

                        self.fd_polygon.shape.vertices = poly

                        t = self.world.CreateStaticBody(
                            fixtures=self.fd_polygon)
                        t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                        self.terrain.append(t)
                    counter = stair_steps * stair_width + 1

            elif state == self.STAIRS and not oneshot:
                s = stair_steps * stair_width - counter
                n = s // stair_width
                y = original_y + (n * stair_height * stair_slope) * TERRAIN_STEP - \
                    (stair_height if stair_slope == -1 else 0) * TERRAIN_STEP


            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == self.GRASS and hardcore:
                    state = self.np_random.integers(1, self._STATES_)
                    oneshot = True
                else:
                    state = self.GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0
        print("Resetting BipedalWalker with: ", self.config)

        self._generate_terrain(self.hardcore)
        if self.render_mode is not None:
            # clouds are just a decoration, with no impact on the physics, so don't add them when not rendering
            self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        self.legs: List[Box2D.b2Body] = []
        self.joints: List[Box2D.b2RevoluteJoint] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        vector_reward = np.zeros(3, dtype=np.float32)
        original_reward = 0
        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        original_shaping = shaping - 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        if self.prev_shaping is not None:
            # we separate balance reward from forward reward in multi-objective setting
            original_reward = original_shaping - self.original_prev_shaping
            vector_reward[0] = shaping - self.prev_shaping
        self.prev_shaping = shaping
        self.original_prev_shaping = shaping

        # Reward for balance (keeping hull angle close to zero)
        max_angle = np.pi / 4  # Maximum angle before considering the walker has fallen
        angle_proportion = max(0, (max_angle - abs(state[0])) / max_angle)
        vector_reward[1] = angle_proportion * 1  # Rescaled to a range of [0, 1]

        for a in action:
            original_reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # use 0.003125 instead of 0.00035 => 4 (number of actions) * -0.003125 * 80 (default motors torque) = -1
            vector_reward[2] -= 0.003125 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        terminated = False
        if self.game_over or pos[0] < 0:
            original_reward = -100
            vector_reward[0] = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), vector_reward, terminated, False, {"original_reward": original_reward}