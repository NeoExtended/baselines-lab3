from abc import ABC

import numpy as np
from typing import Dict, Tuple, Optional


class StepModifier(ABC):
    def __init__(self, action_map: Dict[int, Tuple[int, int]], **kwargs):
        self.maze = None  # type: Optional[np.ndarray]
        self.freespace = None  # type: Optional[np.ndarray]
        self.action_map = action_map
        self.np_random = np.random.random.__self__

    def seed(self, np_random: np.random.Generator):
        self.np_random = np_random

    def reset(self, locations: np.ndarray, maze: np.ndarray, freespace: np.ndarray):
        self.maze = maze
        self.freespace = freespace

    def step(self, action: int, locations: np.ndarray) -> np.ndarray:
        pass

    def step_done(self, valid_locations) -> None:
        pass


class SimpleMovementModifier(StepModifier):
    def step(self, action: int, locations: np.ndarray) -> np.ndarray:
        dy, dx = self.action_map[action]
        return np.array([dy, dx])


class RandomMovementModifier(StepModifier):
    def __init__(self, action_map: Dict[int, Tuple[int, int]], random_move_chance: float = 0.25, random_move_distance: int = 1, **kwargs):
        super(RandomMovementModifier, self).__init__(action_map, **kwargs)
        self.random_move_chance = random_move_chance
        self.random_move_distance = random_move_distance

    def step(self, action: int, locations: np.ndarray) -> np.ndarray:
        if self.random_move_chance > 0.0:
            random_moves = self.np_random.randint(-self.random_move_distance, self.random_move_distance + 1, self.maze.shape + (2,))
            #random_moves_y = self.np_random.randint(0, self.random_move_distance + 1, self.maze.shape)
            mask = self.np_random.choice([0, 1], self.maze.shape + (1,), p=[1 - self.random_move_chance, self.random_move_chance])
            random_moves = random_moves * mask

            return np.squeeze((np.array([random_moves[tuple(locations.T)]])))
        else:
            return np.zeros(locations.shape, dtype=locations.dtype)


class FuzzyMovementModifier(StepModifier):
    def __init__(self, action_map: Dict[int, Tuple[int, int]], fuzzy_action_probability: float = 0.25, **kwargs):
        super(FuzzyMovementModifier, self).__init__(action_map, **kwargs)
        self.fuzzy_action_probability = fuzzy_action_probability

    def step(self, action: int, locations: np.ndarray) -> np.ndarray:
        if self.fuzzy_action_probability == 0.0:
            dy, dx = self.action_map[action]
            return np.array([dx, dy])
        else:
            dy, dx = self.action_map[action]
            global_mask = self.np_random.choice([0, 1], self.maze.shape + (1,),
                                                p=[self.fuzzy_action_probability, 1 - self.fuzzy_action_probability])
            particle_mask = global_mask[tuple(locations.T)]
            return np.where(particle_mask, [dy, dx], [0, 0])


class PhysicalMovementModifier(StepModifier):
    def __init__(self, action_map: Dict[int, Tuple[int, int]], fuzzy_action_probability: float = 0.1, **kwargs):
        super(PhysicalMovementModifier, self).__init__(action_map, **kwargs)

        self.random_particle_weights = False
        self.particle_speed = None
        self.exact_locations = None
        self.force = 0.25
        self.acceleration = None

    def reset(self, locations: np.ndarray, maze: np.ndarray, freespace: np.ndarray):
        super(PhysicalMovementModifier, self).reset(locations, maze, freespace)
        self.exact_locations = np.copy(locations)

        n_particles = len(locations)

        self.particle_speed = np.zeros((n_particles, 2))
        self.exact_locations = np.zeros((n_particles, 2))
        if self.random_particle_weights:
            self.particle_weight = self.np_random.randint(1, 2, size=(n_particles,))
        else:
            self.particle_weight = np.full((n_particles,), 1)

        # self.acceleration = 0.25
        self.acceleration = self.force / self.particle_weight

    def step(self, action: int, locations: np.ndarray) -> np.ndarray:
        dy, dx = self.action_map[action]

        self.particle_speed = self.particle_speed + (np.stack([dy * self.acceleration, dx * self.acceleration], axis=1))
        rounded_update = np.rint(self.particle_speed).astype(int)
        return rounded_update

    def step_done(self, valid_locations):
        self.exact_locations = np.where(valid_locations, self.exact_locations + self.particle_speed, self.exact_locations)
        self.particle_speed = self.particle_speed / 1.1  # drag
        self.particle_speed = np.where(valid_locations, self.particle_speed, self.particle_speed / 2)  # collision
        self.particle_speed = np.where(np.abs(self.particle_speed) > 0.01, self.particle_speed, 0)  # stop really slow particles.