from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm
from baselines_lab3.algorithms.utils.distance_map import DistanceMap

class TargetPointMoverAlgorithm(ParticleMovingAlgorithm):
    def __init__(self, env, goal, simulate=False):
        super().__init__(env, simulate)
        self._costmap = DistanceMap(env, goal)

    def _get_direction(self, coord):
        n = min(self._env.get_neighbored_coords(coord), key=lambda x: self._costmap.distance(x))
        v = self._costmap.distance(n)
        if v >= self._costmap.distance(coord):
            return None
        return (n[0] - coord[0], n[1] - coord[1])

    def _move_closer(self):
        p = self._find_particle()
        if self._costmap.distance(p) <= 0:
            assert self._costmap.distance(p) == 0
            return None
        d = self._get_direction(p)
        self._move(d)
        return d

    def _find_particle(self):
        particles = self.get_particles()
        assert (particles == particles[0]).all(), "Only one particle should be left"
        return tuple(particles[0])

    def run(self):
        assert self.get_environment() is not None and self._costmap is not None, "Necessary data"
        while self._move_closer():
            pass
