from baselines_lab3.algorithms.gym_maze_wrapper import GymMazeWrapper


class ParticleMovingAlgorithm:
    def __init__(self, env: GymMazeWrapper, simulate=False):
        self._env = env
        self._movements = []
        self._movement_callback = []
        self.simulate = simulate

    def add_movement_callback(self, callback):
        self._movement_callback.append(callback)

    def remove_movement_callback(self, callback):
        self._movement_callback.remove(callback)

    def set_environment(self, env):
        self._env = env

    def get_environment(self):
        return self._env

    def get_particles(self):
        return self._env.get_particle_locations()

    def reset_and_run(self):
        self._movements = []
        self.run()

    def run(self):
        raise NotImplementedError()

    def _run_and_imitate_alg(self, alg):
        def copy_moves(other_alg):
            self._move(other_alg.get_movements()[-1])
        alg.add_movement_callback(copy_moves)
        alg.reset_and_run()
        alg.remove_movement_callback(copy_moves)

    def _move(self, dir):
        if type(dir) is list:
            for d in dir:
                self._move(d)
        else:
            if not self.simulate:
                self._env.step(dir)
            self._movements.append(dir)
            for cb in self._movement_callback:
                cb(self)

    def get_movements(self):
        return self._movements

    def get_number_of_movements(self):
        return len(self._movements) if self._movements is not None else None

    def __str__(self):
        return self.__class__.__name__


class NullAlgorithm(ParticleMovingAlgorithm):
    def run(self):
        return self.get_particles()