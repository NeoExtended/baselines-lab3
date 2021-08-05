from algorithms.gathering.iterative_merging import \
    IterativeMostDistancedPairMergingAlgorithm
from algorithms.gathering.local_optimization.minimize_sum_to_closest_extreme import MinimizeSumToClosestExtremeAlgorithm
from algorithms.gathering.pair_merging.merge_via_extreme import MergeViaExtremeAlgorithm
from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm
from algorithms.utils.extremes import Objective

from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper

def OriginalMoveToMinSumExtremumAlgorithm(env: GymMazeWrapper):
    alg = IterativeMostDistancedPairMergingAlgorithm(env)
    alg.set_merging_algorithm(MergeViaExtremeAlgorithm(env))
    alg.add_local_optimization(MinimizeSumToClosestExtremeAlgorithm(env))
    return alg

def OriginalMoveToExtremeAlgorithm(env: GymMazeWrapper):
    alg = IterativeMostDistancedPairMergingAlgorithm(env)
    alg.set_merging_algorithm(MergeViaExtremeAlgorithm(env))
    return alg


class MoveToExtremeAlgorithm(ParticleMovingAlgorithm):
    def __init__(self, env: GymMazeWrapper):
        super().__init__(env)
        self._extreme_comperator = lambda a, b: a < b
        self._optimization = False

    def set_optimization(self, val: bool):
        self._optimization = val

    def set_extreme_comperator(self, l):
        self._extreme_comperator = l

    def _find_extreme(self):
        ex = None
        for i in range(self._env.matrix.shape[0]):
            for j in range(self._env.matrix.shape[1]):
                if self._env.is_valid_position((i, j)):
                    if not ex or self._extreme_comperator((i, j), ex):
                        ex = (i, j)
        return ex

    def _get_extreme_particle(self):
        ex = None
        for particle in self.get_particles():
            if not ex or not self._extreme_comperator(list(particle), ex):
                ex = list(particle)
        assert ex, "Should always be found if there are particles"
        return ex

    def _local_opt_step(self, objective):
        current_obj_value = objective(self.get_particles())
        options = [(self._env.simulate_action(op), op) for op in self.get_environment().get_operations()]
        next = min(options, key=lambda x: objective(x[0]))
        next_obj_val = objective(next[0])

        if next_obj_val < current_obj_value:
            self._move(next[1])
            return next[1]
        else:
            return None

    def run(self):
        extreme_target = self._find_extreme()
        tracked_particle = None
        objective = Objective(self._env, extreme_target)
        while True:
            if tracked_particle == extreme_target:
                tracked_particle = None
            if not tracked_particle:
                if self._optimization:
                    while self._local_opt_step(objective):
                        pass
                tracked_particle = self._get_extreme_particle()
            next_op = objective.get_direction(tracked_particle)
            if not next_op:
                return
            tracked_particle = (tracked_particle[0] + next_op[0],
                                tracked_particle[1] + next_op[1])
            self._move(next_op)

    def __str__(self):
        if self._optimization:
            return str(self.__class__.__name__) + "_opt"
        else:
            return str(self.__class__.__name__) + "_noopt"
