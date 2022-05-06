import copy
import distutils.spawn
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import optuna
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.callbacks import EveryNTimesteps
import yaml

import baselines_lab3.utils.env_util
from baselines_lab3.env import create_environment
from baselines_lab3.env.wrappers import (
    EvaluationWrapper,
    VecEvaluationWrapper,
    VecImageRecorder,
)
from baselines_lab3.experiment import Runner, HyperparameterOptimizer
from baselines_lab3.experiment.samplers import Sampler
from baselines_lab3.model import create_model
from baselines_lab3.model.callbacks import (
    CheckpointManager,
    TensorboardLogger,
    RenderLogger,
)
from baselines_lab3.utils import util, config_util
from baselines_lab3.utils.tensorboard import (
    Plotter,
    TensorboardLogReader,
    EvaluationLogReader,
)

PLOT_TAGS = [
    "episode_length/ep_length_mean",
    "episode_length/eval_ep_length_mean",
    "episode_reward",
    "reward/ep_reward_mean",
    "reward/eval_ep_reward_mean",
]
PLOT_NAMES = [
    "normalized extrinsic reward",
    "intrinsic reward",
    "episode length",
    "eval episode length",
    "total episode reward",
    "episode reward",
    "eval episode reward",
]


class Session(ABC):
    """
    The main experiment control unit. Creates the environment and model from the lab configuration, runs the experiment
        and controls model saving.
    :param config: (dict) The lab configuration.
    :param log_dir: (Path) Path to the logging directory
    """

    def __init__(self, config: Dict, log_dir: Path):
        self.config = config
        self.log = log_dir

    @abstractmethod
    def run(self):
        """
        Starts the experiment.
        """
        pass

    @staticmethod
    def create_session(config: Dict, log_dir: Path):
        lab_mode = config["args"].get("lab_mode", "train")

        if lab_mode == "train":
            return TrainSession(config, log_dir)
        elif lab_mode == "enjoy":
            return ReplaySession(config, log_dir)
        elif lab_mode == "search":
            return SearchSession(config, log_dir)
        else:
            raise ValueError("Unknown lab mode!")

    def _plot(self, log_dir):
        file_format = "pdf"
        if isinstance(self.config["meta"]["plot"], dict):
            file_format = self.config["meta"]["plot"].get("format", file_format)
            PLOT_TAGS.extend(self.config["meta"]["plot"].get("tags"))
            PLOT_NAMES.extend(self.config["meta"]["plot"].get("names"))
        reader = TensorboardLogReader([log_dir])
        plotter = Plotter(log_dir, file_format=file_format)
        plotter.from_reader(reader, PLOT_TAGS, PLOT_NAMES)


class ReplaySession(Session):
    """
    Control unit for a replay session (enjoy lab mode - includes model evaluation).
    """

    def __init__(self, config: Dict, log_dir: Path):
        super(ReplaySession, self).__init__(config, log_dir)

        args = SimpleNamespace(**config["args"])
        if args.checkpoint_path:
            self.data_path = args.checkpoint_path
        else:
            self.data_path = os.path.split(
                os.path.dirname(config["algorithm"]["trained_agent"])
            )[0]

        self.env = create_environment(
            config=config,
            seed=self.config["meta"]["generated_seed"],
            log_dir=self.data_path,
            video_path=self.data_path if args.obs_video else None,
            evaluation=args.evaluate,
        )

        self.agent = create_model(
            config["algorithm"],
            self.env,
            self.log,
            seed=self.config["meta"]["generated_seed"],
        )
        other_agents = self._setup_additional_agents(config, args)
        obs = self.env.reset()

        self.deterministic = not args.stochastic
        if args.video:
            self._setup_video_recorder(self.data_path)

        self.evaluate = args.evaluate
        if args.evaluate:
            self.num_episodes = args.evaluate
            self.eval_wrapper = baselines_lab3.utils.env_util.unwrap_env(
                self.env, VecEvaluationWrapper, EvaluationWrapper
            )
            self.eval_wrapper.aggregator.path = self.data_path
        else:
            # Render about 4 complete episodes per env in enjoy mode without evaluation.
            self.num_episodes = self.config["env"]["n_envs"] * 4

        self.runner = Runner(
            self.env,
            self.agent,
            deterministic=self.deterministic,
            other_agents=other_agents,
            random=args.random_agent,
        )

        if args.plot:
            self._plot(config["meta"]["session_dir"])

    def _setup_additional_agents(self, config, args):
        other_agents = None
        if config.get("enjoy", None):
            if config["enjoy"].get("agreement", None):
                other_agents = []
                for agent in config["enjoy"]["agreement"]:
                    cfg = copy.deepcopy(config)
                    config_util.set_checkpoints(cfg, agent, args.type, args.trial)
                    other_agents.append(
                        create_model(
                            cfg["algorithm"],
                            self.env,
                            self.log,
                            seed=self.config["meta"]["generated_seed"],
                        )
                    )
        return other_agents

    def _setup_video_recorder(self, video_path):
        if distutils.spawn.find_executable("avconv") or distutils.spawn.find_executable(
            "ffmpeg"
        ):
            logging.info("Using installed standard video encoder.")
            self.env = VecVideoRecorder(
                self.env,
                video_path,
                record_video_trigger=lambda x: x == 0,
                video_length=10000,
                name_prefix=util.get_timestamp(),
            )
        else:
            logging.warning(
                "Did not find avconf or ffmpeg - using gif as a video container replacement."
            )
            self.env = VecImageRecorder(self.env, video_path)

    def _finish_evaluation(self):
        # Hacky stuff, need to refactor environment evaluation system. :(
        if self.evaluate:
            reader = EvaluationLogReader([self.data_path])
            data = reader.read_data(["max_distance"])[self.data_path]
            steps = []
            for env in data["max_distance"][0]:
                steps.append(env[-1])
            avg_steps = np.average(steps)
            with open(self.eval_wrapper.aggregator.last_save, "r") as f:
                evaluation_results = yaml.load(f)
            evaluation_results["accurate_episode_length"] = float(avg_steps)
            evaluation_results["steps"] = steps
            with open(self.eval_wrapper.aggregator.last_save, "w") as f:
                yaml.dump(evaluation_results, f)

            logging.info("Number of steps pre wrappers: {}".format(avg_steps))

    def run(self):
        self.runner.run(self.num_episodes)
        self._finish_evaluation()


class TrainSession(Session):
    """
    Control unit for the training lab mode.
    """

    def __init__(self, config: Dict, log_dir: Path):
        super(TrainSession, self).__init__(config, log_dir)
        self.env = None
        self.recording_env = None
        self.agent = None
        self.saver = None

        self.record_video = config["args"].get("video")
        self.video_format = config["args"].get("video_format")

    def _setup_session(self):
        self.env = create_environment(
            config=self.config,
            seed=self.config["meta"]["generated_seed"],
            log_dir=str(self.log),
        )
        if self.record_video:
            self._setup_recorder(
                os.path.join(util.get_log_directory(), "videos"), self.video_format
            )

        self.agent = create_model(
            self.config["algorithm"],
            self.env,
            log_dir=self.log,
            seed=self.config["meta"]["generated_seed"],
        )

        save_interval = self.config["meta"].get("save_interval", 250000)
        n_keep = self.config["meta"].get("n_keep", 5)
        keep_best = self.config["meta"].get("keep_best", True)
        n_eval_episodes = self.config["meta"].get("n_eval_episodes", 32)

        self.saver = CheckpointManager(
            model_dir=os.path.join(util.get_log_directory(), "checkpoints"),
            save_interval=save_interval,
            n_keep=n_keep,
            keep_best=keep_best,
            n_eval_episodes=n_eval_episodes,
            config=self.config,
            training_env=self.env,
            tb_log=bool(self.log),
        )

    def _setup_recorder(self, path, format):
        self.env = VecImageRecorder(self.env, path, format=format, unvec=True)

    def run(self):
        self._setup_session()
        self._run_experiment()
        del self.env
        del self.recording_env
        del self.agent
        del self.saver

        if self.config["meta"].get("plot", False):
            self._plot(self.log)

    def _run_experiment(self):
        callbacks = [self.saver, TensorboardLogger(config=self.config)]

        recorder_config = self.config["meta"].get("record_renders", False)
        if recorder_config:
            if isinstance(recorder_config, bool):
                recorder_config = {}
            env_config = copy.deepcopy(self.config)
            env_config["env"]["n_envs"] = recorder_config.pop("n_envs", 4)
            self.recording_env = create_environment(
                config=env_config,
                seed=self.config["meta"]["generated_seed"],
                log_dir=None,
            )
            callbacks.append(
                EveryNTimesteps(
                    n_steps=recorder_config.pop("interval", 10000),
                    callback=RenderLogger(
                        render_env=self.recording_env, **recorder_config, verbose=1
                    ),
                )
            )

        logging.info("Starting training.")
        self.agent.learn(self.config["meta"]["n_timesteps"], callback=callbacks)
        self.env.close()

        if self.recording_env is not None:
            self.recording_env.close()


class SearchSession(Session):
    """
    Control unit for the search lab mode
    """

    def __init__(self, config: Dict, log_dir: Path):
        super(SearchSession, self).__init__(config, log_dir)

        self.optimizer = HyperparameterOptimizer(
            config, self.log, config["args"].get("mail")
        )
        self.plot = config["args"].get("plot")

    def run(self):
        study = self.optimizer.optimize()
        self._log_study_info(study)

        dataframe = study.trials_dataframe()
        dataframe.to_csv(os.path.join(self.log, "search_history.csv"))
        if self.plot:
            # Suppress find font spam
            logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
            try:
                hist = dataframe.hist()
            except:
                logging.warning("Could not plot data due to infinite values. :(")
            plt.show()

        promising = self._find_promising_trials(study)
        for i, trial in enumerate(promising):
            self._save_config(
                study.trials[trial[0]], "promising_trial_{}.yml".format(i)
            )
        self._save_config(study.best_trial, "best_trial.yml")

    def _save_config(self, trial: optuna.Trial, name: str):
        sample = Sampler.create_sampler(self.config).sample(trial)
        config_util.save_config(sample, self.log / name)

    def _find_promising_trials(self, study):
        promising = list()
        for idx, trial in enumerate(study.trials):
            promising.append([idx, trial.value])

        promising = sorted(promising, key=lambda x: x[1])
        return promising[:3]

    def _log_study_info(self, study):
        logging.info(f"Number of finished trials: {len(study.trials)}")
        trial = study.best_trial
        logging.info(f"Best trial: Trial {trial.number}")

        logging.info(f"Value: {trial.value}")
        logging.info("Params: ")
        for key, value in trial.params.items():
            logging.info(f"  {key}: {value}")

        if optuna.visualization.is_available():
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(os.path.join(self.log, "optimization_history.html"))
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(self.log, "param_importances.html"))
        else:
            logging.warning(
                "Please install plotly>=4.0.0 to generate visualizations of the optimization process."
            )
