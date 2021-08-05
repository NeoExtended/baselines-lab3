from algorithms.gathering.iterative_merging import \
    IterativeMostDistancedPairMergingAlgorithm
from algorithms.gathering.pair_merging.merge_via_dynamic_shortest_path import \
    MergeViaDynamicShortestPathAlgorithm
from algorithms.gathering.pair_merging.merge_via_static_shortest_path import \
    MergeViaStaticShortestPathAlgorithm

def DynamicShortestPathFollowingAlgorithm(env):
    alg = IterativeMostDistancedPairMergingAlgorithm(env)
    alg.set_merging_algorithm(MergeViaDynamicShortestPathAlgorithm(env))
    return alg

def StaticShortestPathFollowingAlgorithm(env):
    alg = IterativeMostDistancedPairMergingAlgorithm(env)
    alg.set_merging_algorithm(MergeViaStaticShortestPathAlgorithm(env))
    return alg

