import random

import numpy as np

from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm
from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper


class MoveUntilNoChangeAlgorithm(ParticleMovingAlgorithm):
    def __init__(self, env: GymMazeWrapper, simulate=False):
        super().__init__(env, simulate)
        self._movement_pattern = []

    def set_movement_pattern(self, moves):
        self._movement_pattern = moves

    def run(self):
        while True:
            old_state = self.get_particles()
            self._move(self._movement_pattern)
            if (old_state == self.get_particles()).all():
                return

class MoveToRandomCornerAlgorithm(ParticleMovingAlgorithm):
    _patterns = [[(1, 0), (0, 1)],
                 [(-1, 0), (0, 1)],
                 [(1, 0), (0, -1)],
                 [(-1, 0), (0, -1)]]

    def run(self):
        option = MoveUntilNoChangeAlgorithm(env=self.get_environment())
        option.set_movement_pattern(random.sample(self._patterns, k=1)[0])
        option.reset_and_run()
        self._movements = option.get_movements()
        #self._move(option.get_movements())
        return self.get_particles()


class MoveToCornerWithLeastParticlesAlgorithm(ParticleMovingAlgorithm):
    _patterns = [[(1, 0), (0, 1)],
                 [(-1, 0), (0, 1)],
                 [(1, 0), (0, -1)],
                 [(-1, 0), (0, -1)]]

    def run(self):
        # TODO: Not compatible with current env!
        options = [MoveUntilNoChangeAlgorithm(self.get_environment()) for i in range(len(self._patterns))]
        for i, option in enumerate(options):
            option.set_environment(self.get_environment())
            option.set_movement_pattern(self._patterns[i])
            option.reset_and_run()
        best = min(options, key=lambda x: np.sum(x.get_particles()))
        self._move(best.get_movements())
