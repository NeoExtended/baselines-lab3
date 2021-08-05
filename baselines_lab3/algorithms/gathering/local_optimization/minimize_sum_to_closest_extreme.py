from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm
from algorithms.utils.extremes import EXTREME_COMPERATORS, Objective


class MinimizeSumToClosestExtremeAlgorithm(ParticleMovingAlgorithm):
    def __init__(self, env, simulate=False):
        super().__init__(env, simulate)
        self._extreme_targets = self._find_extremes()

    def _find_extreme(self, cmp):
        ex = None
        for i in range(self._env.matrix.shape[0]):
            for j in range(self._env.matrix.shape[1]):
                if self._env.is_valid_position((i, j)):
                    if not ex or cmp((i, j), ex):
                        ex = (i, j)
        return ex

    def _find_extremes(self):
        return [self._find_extreme(cmp) for cmp in EXTREME_COMPERATORS]

    def _find_best_objective(self, extreme_targets):
        return min(
            (Objective(self.get_environment(), target) for target in extreme_targets),
            key=lambda x: x(self.get_particles()))

    def set_environment(self, env):
        super().set_environment(env)
        self._extreme_targets = self._find_extremes()
        return self

    def _local_opt_step(self, objective):
        current_obj_value = objective(self.get_particles())
        options = [(self._env.simulate_action(op), op) for op in
                   self.get_environment().get_operations()]
        next = min(options, key=lambda x: objective(x[0]))
        next_obj_val = objective(next[0])

        if next_obj_val < current_obj_value:
            self._move(next[1])
            return next[1]
        else:
            return None

    def run(self):
        objective = self._find_best_objective(self._extreme_targets)
        while self._local_opt_step(objective):
            pass
        return self.get_particles()
