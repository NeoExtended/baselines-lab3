from collections import deque

from algorithms.gathering.pair_merging.pair_merging_algorithm import PairMergingAlgorithm
from algorithms.utils.distance_map import DistanceMap


class MergeViaDynamicShortestPathAlgorithm(PairMergingAlgorithm):

    def _compute_shortest_path(self):
        dm = DistanceMap(self._env, self.get_second_particle())
        pos = self.get_first_particle()
        path = deque()
        while pos != self.get_second_particle():
            dir = dm.get_direction(pos)
            path.append(dir)
            pos = self._env.add(pos, dir)
        return path

    def move(self):
        if not self._path:
            return None
        alt_path = self._compute_shortest_path()
        if len(alt_path) < len(self._path):
            self._path = alt_path
        if not self._path:
            return None
        dir = self._path.popleft()
        if self._env.is_valid_position(self._env.add(self._b, dir)):
            last_op = self._path.pop() if self._path else None
            if not last_op or self._env.add(dir, last_op) != (0, 0):
                if last_op:
                    self._path.append(last_op)
                self._path.append(dir)
        return dir

    def remaining_distance(self):
        if len(self._path)==0:
            assert self.get_first_particle() == self.get_second_particle()
        else:
            assert self.get_first_particle() != self.get_second_particle()
        return len(self._path)

    def run(self):
        self._path = self._compute_shortest_path()
        while self.remaining_distance() > 0:
            self._move(self.move())
        return self.get_particles()
