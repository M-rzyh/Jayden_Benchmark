from scipy.special import binom
import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import AbstractLane, LineType
from highway_env.vehicle.behavior import IDMVehicle
import matplotlib.pyplot as plt

bernstein = lambda n, k, t: binom(n, k) * t**k * (1. - t)**(n - k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]

def get_bezier_curve(a=None, rad=0.2, edgy=0, **kw):
    """ Given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    if a is None:
        a = get_random_points(**kw)

    numpoints = kw.get('numpoints', 30)

    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var", numpoints=numpoints)
    x, y = c.T
    return x, y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0, **kw):
    """Create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7 / n
    np_random = kw.get('np_random', np.random)
    a = np_random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n,
                                 scale=scale, mindst=mindst, rec=rec + 1, np_random=np_random)

class BezierLane(AbstractLane):
    def __init__(self, control_points, width=AbstractLane.DEFAULT_WIDTH, line_types=None, speed_limit=20):
        super().__init__()
        self.control_points = control_points
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.speed_limit = speed_limit
        self.curve = bezier(control_points)
        self.length = self._compute_length()

    def _compute_length(self):
        num_samples = 100
        length = 0.0
        prev_point = self.position(0)
        for i in range(1, num_samples + 1):
            t = i / num_samples
            point = self.position(t)
            length += np.linalg.norm(point - prev_point)
            prev_point = point
        return length

    def position(self, t):
        return self.curve[int(t * (len(self.curve) - 1))]

    def heading_at(self, t):
        p0 = self.position(t)
        p1 = self.position(min(t + 0.01, 1))
        return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

    def local_coordinates(self, position):
        num_samples = 100
        closest_t = 0
        closest_distance = float('inf')
        for i in range(num_samples + 1):
            t = i / num_samples
            point = self.position(t)
            distance = np.linalg.norm(position - point)
            if distance < closest_distance:
                closest_distance = distance
                closest_t = t
        lateral = np.dot(position - self.position(closest_t), [-np.sin(self.heading_at(closest_t)), np.cos(self.heading_at(closest_t))])
        return closest_t, lateral

    def width_at(self, t):
        return self.width