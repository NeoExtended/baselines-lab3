from algorithms.gathering.pair_merging.pair_merging_algorithm import PairMergingAlgorithm
from algorithms.utils.distance_map import DistanceMap
from algorithms.utils.extremes import EXTREME_COMPERATORS


class MergeViaExtremeAlgorithm(PairMergingAlgorithm):

    def _make_a_extreme(self, extreme_cmp):
        if extreme_cmp(self._a, self._b):
            self.swap_pair()

    def _find_extreme(self, extreme_cmp):
        ex = None
        for i in range(self._env.matrix.shape[0]):
            for j in range(self._env.matrix.shape[1]):
                if self._env.is_valid_position((i, j)):
                    if not ex or extreme_cmp((i, j), ex):
                        ex = (i, j)
        return ex

    def _find_best_extreme(self):
        extreme_options = [(DistanceMap(self.get_environment(),
                                        self._find_extreme(extreme_cmp)), extreme_cmp)
                           for extreme_cmp in EXTREME_COMPERATORS]
        best_option = min(extreme_options,
                          key=lambda x: x[0].distance(self._a) + x[0].distance(
                              self._b))
        return best_option[0], best_option[1]

    def run(self):
        while not self.are_merged():
            dm, extreme_cmp = self._find_best_extreme()
            self._make_a_extreme(extreme_cmp)
            while not self.are_merged() and dm.distance(self._a) > 0:
                self._move(dm.get_direction(self._a))

    def __str__(self):
        return self.__class__.__name__
