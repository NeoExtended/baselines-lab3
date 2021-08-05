from gym.envs.registration import register

from baselines_lab3.env.gym_maze.maze_generators import BufferedRRTGenerator, StagesRRTGenerator

register(id="Maze0122Discrete-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122.csv',
                 'goal': [96, 204],
                 'goal_range': 15,
                 'reward_generator': "goal"})

register(id="Maze0122Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122.csv',
                 'goal': [96, 204],
                 'goal_range': 15,
                 'reward_generator': "continuous"})

register(id="Maze0122Continuous-v1",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122.csv',
                 'goal': [96, 204],
                 'goal_range': 15,
                 'n_particles': -1,
                 'reward_generator': "continuous"})

register(id="Maze0122Continuous-v2",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122.csv',
                 'goal': None,
                 'goal_range': 15,
                 'reward_generator': "continuous"})

register(id="Maze0122SmallContinuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122_small.csv',
                 'goal': [48, 102],
                 'goal_range': 15,
                 'reward_generator': "continuous"})

register(id="Maze0122TinyContinuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122_small.csv',
                 'goal': [28, 59],
                 'goal_range': 15,
                 'reward_generator': "continuous"})

register(id="Maze0318Discrete-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': [82, 80],
                 'goal_range': 10,
                 'reward_generator': "goal"})

register(id="Maze0318Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': [82, 80],
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="Maze0318Continuous-v1",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': [82, 80],
                 'goal_range': 10,
                 'n_particles': -1,
                 'reward_generator': "continuous"})

register(id="Maze0318Continuous-v2",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': None,
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="Maze0318Continuous-v3",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': None,
                 'goal_range': 10,
                 'n_particles': -1,
                 'reward_generator': "continuous"})

register(id="Maze0318Continuous-v4",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': [82, 80],
                 'goal_range': 10,
                 'reward_generator': "continuous",
                 'step_type': "fuzzy"})

register(id="Maze0518Discrete-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0518.csv',
                 'goal': [61, 130],
                 'goal_range': 10,
                 'reward_generator': "goal"})

register(id="Maze0518Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0518.csv',
                 'goal': [61, 130],
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="Maze0518Continuous-v1",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0518.csv',
                 'goal': [61, 130],
                 'goal_range': 10,
                 'n_particles': -1,
                 'reward_generator': "continuous"})

register(id="Maze0518Continuous-v2",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0518.csv',
                 'goal': None,
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="RandomMazeContinuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': BufferedRRTGenerator,
                 'goal': None,
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="RandomMazeDiscrete-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': BufferedRRTGenerator,
                 'goal': None,
                 'goal_range': 10,
                 'reward_generator': "goal"})

register(id="StagesMazeContinuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': StagesRRTGenerator,
                 'goal': None,
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="PhysicalMaze0318Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0318.csv',
                 'goal': [82, 80],
                 'goal_range': 10,
                 'reward_generator': "continuous",
                 'step_type': "physical"})

register(id="PhysicalVesselMaze02Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel_2.csv',
                 'goal': [85, 66],
                 'goal_range': 10,
                 'reward_generator': "continuous",
                 'step_type': "physical"})

register(id="PhysicalMaze0122Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/map0122.csv',
                 'goal': [96, 204],
                 'goal_range': 15,
                 'reward_generator': "continuous",
                 'step_type': "physical"})

register(id="VesselMaze01Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel.csv',
                 'goal': [95, 60],
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="VesselMaze01Discrete-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel.csv',
                 'goal': [95, 60],
                 'goal_range': 10,
                 'reward_generator': "goal"})

register(id="VesselMaze02Continuous-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel_2.csv',
                 'goal': [85, 66],
                 'goal_range': 10,
                 'reward_generator': "continuous"})

register(id="VesselMaze02Discrete-v0",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel_2.csv',
                 'goal': [85, 66],
                 'goal_range': 10,
                 'reward_generator': "goal"})

register(id="VesselMaze02Continuous-v1",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel_2.csv',
                 'goal': [85, 66],
                 'goal_range': 10,
                 'n_particles': -1,
                 'reward_generator': "continuous"})

register(id="VesselMaze02Continuous-v2",
         entry_point="baselines_lab.env.gym_maze.envs:MazeBase",
         max_episode_steps=2000,
         kwargs={'instance': '../mapdata/small_vessel_2.csv',
                 'goal': None,
                 'goal_range': 10,
                 'reward_generator': "continuous"})
