from algorithms.gathering.pair_merging.pair_merging_algorithm import PairMergingAlgorithm
from algorithms.gathering.pair_merging.pair_selector import \
    MostDistancedPairFinderAlgorithm
from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm


def IterativeMostDistancedPairMergingAlgorithm(env):
    alg = IterativePairMergingAlgorithm(env)
    alg.set_pair_selector(MostDistancedPairFinderAlgorithm(env))
    return alg


class IterativePairMergingAlgorithm(ParticleMovingAlgorithm):
    def __init__(self, env):
        super().__init__(env)
        self._merging_algorithm = None
        self._local_optimization_algorithms = []

    def set_merging_algorithm(self, algorithm: PairMergingAlgorithm):
        self._merging_algorithm = algorithm
        self._merging_algorithm.simulate = True
        return self

    def set_pair_selector(self, algorithm):
        self._pair_selector = algorithm
        return self

    def add_local_optimization(self, algorithm):
        algorithm.simulate = True
        self._local_optimization_algorithms.append(algorithm)
        return self

    def remove_local_optimization(self, algorithm):
        self._local_optimization_algorithms.remove(algorithm)
        return self

    def run(self):
        while True:
            for alg in self._local_optimization_algorithms:
                self._run_and_imitate_alg(alg)
            a, b = self._pair_selector(self.get_particles())
            if a == b:
                return self.get_particles()
            else:
                self._merging_algorithm.set_particle_pair(a, b)
                self._run_and_imitate_alg(self._merging_algorithm)

    def __str__(self):
        if self._local_optimization_algorithms:
            local_opt = "local_optimization=["
            for alg in self._local_optimization_algorithms:
                local_opt += str(alg) + ", "
            local_opt = local_opt[:-2] + "]"
            return self.__class__.__name__ + "<" + str(self._pair_selector) + "," + str(
                self._merging_algorithm) + "," + local_opt + ">"
        else:
            return self.__class__.__name__ + "<" + str(self._pair_selector) + "," + str(
                self._merging_algorithm) + ">"
