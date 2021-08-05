from collections import deque

import numpy as np

from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper


class DistanceMap:
    def __init__(self, env: GymMazeWrapper, target):
        self._env = env
        self._target = target
        self._compute_distance_map()

    def distance(self, coord):
        return self._distances[tuple(coord)]

    def get_distance_matrix(self):
        return self._distances

    def get_direction(self, coord):
        n = min(self._env.get_neighbored_coords(coord), key=lambda x: self.distance(x))
        v = self.distance(n)
        if v < self.distance(coord):
            return (n[0] - coord[0], n[1] - coord[1])

    def _compute_distance_map(self):
        self._distances = np.full(self._env.matrix.shape, np.inf)
        self._distances[self._target] = 0
        queue = deque()
        for n in self._env.get_neighbored_coords(self._target):
            queue.append(n)
        while queue:
            c = queue.popleft()
            current_value = self.distance(c)
            new_val = min(self.distance(n) + 1 for n in self._env.get_neighbored_coords(c))
            if new_val < current_value:
                self._distances[c] = new_val
                for n in self._env.get_neighbored_coords(c):
                    queue.append(n)


