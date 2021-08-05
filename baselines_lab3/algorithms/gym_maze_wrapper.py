from typing import Tuple

import gym
import numpy as np

from baselines_lab3.env.gym_maze.envs.maze_base import MazeBase


class GymMazeWrapper(gym.Wrapper):
    """
    This class works as an adapter to make the gym environment MazeBase compatible for the gathering algorithms.
    """
    def __init__(self, env: MazeBase, allow_diagonal: bool = False, render: bool = False):
        super(GymMazeWrapper, self).__init__(env)
        self.env.reset()
        self.render = render

        if allow_diagonal:
            self._allowed_operations = [(1,0), (-1,0), (0, 1), (0, -1), (-1,-1), (1,1),(-1,1), (1,-1)]
        else:
            self._allowed_operations = [(1,0), (-1,0), (0, 1), (0, -1)]

    def is_valid_position(self, coord):
        row, column = coord
        m = self.env.freespace
        if not 0 <= row < m.shape[0] or not 0 <= column < m.shape[1]:
            return False
        return m[coord] == 1

    def get_neighbored_coords(self, coord: Tuple[int, int]):
        nbrs = []
        for op in self.get_operations():
            n = (coord[0] + op[0], coord[1] + op[1])
            if n != coord and self.is_valid_position(n):
                nbrs.append(n)
        return nbrs

    def get_particle_locations(self):
        return self.env.particle_locations

    def get_operations(self):
        return self._allowed_operations

    def size(self):
        return np.sum(self.env.freespace)

    def step(self, action):
        if self.render:
            self.env.render(mode="human")
        return self.env.step(self.action_to_number(action))

    def action_to_number(self, action):
        return self.env.rev_action_map[action]

    def add(self, coord, dir):
        return (coord[0]+dir[0], coord[1]+dir[1])

    @property
    def matrix(self):
        return self.env.freespace

    def simulate_action(self, action, particle_locations=None):
        dy, dx = action

        if not particle_locations:
            particle_locations = self.env.particle_locations

        new_locations = particle_locations + [dy, dx]
        valid_locations = (self.env.freespace.ravel()[(new_locations[:, 1] + new_locations[:, 0] * self.env.freespace.shape[1])]).reshape(-1, 1)  # Border does not need to be checked as long as all maps have borders.
        return np.where(valid_locations, new_locations, particle_locations)

    def simulate_particle_move(self, particle, direction):
        new_location = particle + np.array(direction)

        if self.env.freespace[tuple(new_location)]:
            return tuple(new_location)
        return particle

    def get_goal(self):
        # Translate goal position from x, y to y, x
        return [self.env.goal[1], self.env.goal[0]]
