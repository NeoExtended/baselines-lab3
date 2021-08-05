from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm
from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper


class PairMergingAlgorithm(ParticleMovingAlgorithm):
    def __init__(self, env: GymMazeWrapper, simulate=False):
        super().__init__(env, simulate)
        self._a = None
        self._b = None

        def update_particle_positions(alg: ParticleMovingAlgorithm):
            last_move = alg.get_movements()[-1]
            self._a = alg.get_environment().simulate_particle_move(self._a, last_move)
            self._b = alg.get_environment().simulate_particle_move(self._b, last_move)
        self.add_movement_callback(update_particle_positions)

    def get_first_particle(self):
        return self._a

    def get_second_particle(self):
        return self._b

    def swap_pair(self):
        self._a, self._b = self._b, self._a

    def set_particle_pair(self, a, b):
        self._a = tuple(a)
        self._b = tuple(b)

    def are_merged(self):
        return self._a == self._b
