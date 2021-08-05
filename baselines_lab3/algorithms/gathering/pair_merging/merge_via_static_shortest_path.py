from algorithms.gathering.pair_merging.pair_merging_algorithm import PairMergingAlgorithm
from algorithms.utils.distance_map import DistanceMap

class MergeViaStaticShortestPathAlgorithm(PairMergingAlgorithm):
    def run(self):
        while self.get_first_particle() != self.get_second_particle():
            dm = DistanceMap(self.get_environment(), self.get_second_particle())
            while dm.distance(self.get_first_particle())>0:
                self._move(dm.get_direction(self.get_first_particle()))

    def __str__(self):
        return self.__class__.__name__