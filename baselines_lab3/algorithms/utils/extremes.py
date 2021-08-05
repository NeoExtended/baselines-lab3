import numpy as np

from algorithms.utils.distance_map import DistanceMap
from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper

__coord_transformers = [lambda c: c,
                        lambda c: (-c[0], c[1]),
                        lambda c: (c[0], -c[1]),
                        lambda c: (-c[0], -c[1]),
                        lambda c: (c[1], c[0]),
                        lambda c: (-c[1], c[0]),
                        lambda c: (c[1], -c[0]),
                        lambda c: (-c[1], -c[0])]
EXTREME_COMPERATORS = [lambda a, b: t(a) > t(b) for t in __coord_transformers]


class Objective:
    def __init__(self, env: GymMazeWrapper, extreme):
        self._env = env
        self._extreme = extreme
        self._distance_map = DistanceMap(self._env, self._extreme)
        #np.seterr(invalid='ignore') # for multiplying np.inf with 0

    def distance(self, coord):
        return self._distance_map.distance(coord)

    def get_direction(self, coord):
        return self._distance_map.get_direction(coord)

    def __call__(self, particles):
        map = self._distance_map.get_distance_matrix()
        cost = map.ravel()[(particles[:, 1] + particles[:, 0] * map.shape[1])]
        total_cost = np.sum(cost)

        #val = np.sum(np.nan_to_num(particles*self._distance_map.get_distance_matrix()))
        return total_cost

