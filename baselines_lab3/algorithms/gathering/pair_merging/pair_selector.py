import random

from algorithms.utils.distance_map import DistanceMap
from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper


class PairFinderAlgorithm:
    def __init__(self, env: GymMazeWrapper):
        self._env = env

    def __call__(self, particles):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__


class MostDistancedPairFinderAlgorithm(PairFinderAlgorithm):
    def __init__(self, env: GymMazeWrapper):
        super(MostDistancedPairFinderAlgorithm, self).__init__(env)

    def _arbitrary_particle(self, particles):
        return tuple(particles[0])

    def _max_distanced_particle(self, particles, p):
        dm = DistanceMap(self._env, p)
        max_p = max(particles, key=lambda x: dm.distance(x))
        return tuple(max_p), dm.distance(max_p)

    def __call__(self, particles):
        a = self._arbitrary_particle(particles)
        b, dist = self._max_distanced_particle(particles, a)
        a = b
        b, dist = self._max_distanced_particle(particles, a)
        return a, b


class RandomPairFinderAlgorithm(PairFinderAlgorithm):

    def __init__(self, env: GymMazeWrapper):
        super(RandomPairFinderAlgorithm, self).__init__(env)

    def __call__(self, particles):
        if len(particles) == 1:
            return particles[0], particles[0]
        assert len(particles) > 0
        pair = random.sample(particles, k=2)
        return pair[0], pair[1]
