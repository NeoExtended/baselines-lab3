import copy
import logging
import os
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from baselines_lab3.env.evaluation import Evaluator
from baselines_lab3.utils import util


class CheckpointManager(BaseCallback):
    """
    Class to manage model checkpoints.
    :param model_dir: (str) Target directory for all the checkpoints.
    :param save_interval: (int) Interval at which models will be saved. Note that the real interval may depend on the
        frequency of calls to the step() function.
    :param n_keep: Number of models to keep. when saving the newest model n+1 the oldest will be deleted automatically.
    :param n_eval_episodes: Number of episodes used for model evaluation.
    :param keep_best: Whether or not to also save the best model. The best model is determined by running a test each
        time a new model is saved. This may take some time.
    :param config: Current lab configuration. Needed to create an evaluation environment if keep_best=True
    :param env: If the lab environment uses a running average normalization like VecNormalize, the running averages of
        the given env will be saved along with the model.
    :param tb_log: Set to true if the evaluation results should be logged. (Only works with keep_best=True)
    """
    def __init__(self, model_dir, save_interval=250000, n_keep=5, keep_best=True, n_eval_episodes=32, eval_method="normal",
                 config=None, env=None, tb_log=False, verbose=0):
        super(CheckpointManager, self).__init__(verbose)
        self.model_dir = model_dir
        self.save_interval = save_interval
        self.n_keep = n_keep
        self.keep_best = keep_best
        self.n_eval_episodes = n_eval_episodes

        if keep_best or env:
            assert config, "You must provide an environment configuration to evaluate the model!"

        self.last_models = []
        self.best = None
        self.best_score = float('-inf')
        self.last_save = 0
        self.wrappers = []
        self.tb_log = tb_log
        self.writer = None

        if config:
            env_desc = copy.deepcopy(config['env'])
            normalization = env_desc.get('normalize', False)
            curiosity = env_desc.get('curiosity', False)

            if normalization:
                self.wrappers.append(('normalization', 'pkl', util.unwrap_vec_env(env, VecNormalize)))
                # Remove unnecessary keys
                if isinstance(normalization, dict):
                    normalization.pop('precompute', None)
                    normalization.pop('samples', None)

            self.evaluator = Evaluator(config,
                                       env=env,
                                       n_eval_episodes=n_eval_episodes,
                                       deterministic=False,
                                       eval_method=eval_method)

    def close(self):
        self.evaluator.close()
    
    def _on_training_start(self) -> None:
        self.writer = self.locals['writer']
    
    def _on_step(self) -> bool:
        if self.num_timesteps >= self.last_save + self.save_interval:
            self.last_save = self.num_timesteps
            self.save(self.model)
        return True
    
    def _on_training_end(self) -> None:
        self.last_save = self.num_timesteps
        self.save(self.model)

    def save(self, model):
        """
        Explicitly saves the given model.
        :param model: stable-baselines model.
        """
        if self.n_keep > 0:
            self._save_model(model)
        if self.keep_best:
            reward, steps = self._save_best_model(model)
            self._log(reward, steps)

    def _save_model(self, model):
        logging.info("Saving last model at timestep {}".format(str(self.num_timesteps)))
        checkpoint = self._checkpoint(self.model_dir, "", self.num_timesteps, util.get_timestamp())
        self._create_checkpoint(checkpoint, model)
        self.last_models.append(checkpoint)

        if len(self.last_models) > self.n_keep:
            old_checkpoint = self.last_models[0]
            del self.last_models[0]
            self._remove_checkpoint(old_checkpoint)

    def _make_path(self, checkpoint, name, extension="zip"):
        return os.path.join(self.model_dir,
                            self._build_filename(checkpoint, name, extension=extension))

    @staticmethod
    def _build_filename(checkpoint, prefix, extension="zip"):
        suffix = checkpoint['type']
        if len(suffix) > 0:
            return "{}_{}_{}_{}.{}".format(prefix, checkpoint['counter'], checkpoint['time'], suffix, extension)
        else:
            return "{}_{}_{}.{}".format(prefix, checkpoint['counter'], checkpoint['time'], extension)

    def _save_best_model(self, model):
        logging.debug("Evaluating model.")
        reward, steps = self.evaluator.evaluate(model)
        logging.debug("Evaluation result: Avg reward: {:.4f}, Avg Episode Length: {:.2f}".format(reward, steps))

        if reward > self.best_score:
            logging.info("Found new best model with a mean reward of {:.4f}".format(reward))
            self.best_score = reward
            if self.best:
                self._remove_checkpoint(self.best)

            self.best = self._checkpoint(self.model_dir, "best", self.num_timesteps, util.get_timestamp())
            self._create_checkpoint(self.best, model)

        return reward, steps

    def _log(self, reward, steps):
        if not self.tb_log:
            return

        self.logger.record('episode_length/eval_ep_length_mean', steps)
        self.logger.record('reward/eval_ep_reward_mean', reward)

    def _remove_checkpoint(self, checkpoint):
        model_path = self._make_path(checkpoint, "model", extension="zip")
        os.remove(model_path)

        for wrapper in self.wrappers:
            wrapper_path = self._make_path(checkpoint, wrapper[0], extension=wrapper[1])
            os.remove(wrapper_path)

    def _create_checkpoint(self, checkpoint, model):
        model_path = self._make_path(checkpoint, "model", extension="zip")
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        model.save(model_path)

        for wrapper in self.wrappers:
            save_path = self._make_path(checkpoint, wrapper[0], extension=wrapper[1])
            wrapper[2].save(save_path)

    @classmethod
    def get_checkpoint(cls, path: str, type: str = "best", trial: int = -1) -> dict:
        """
        Returns a dictionary defining a model checkpoint. The checkpoint may contain more than one archive for different objects.
        :param path: (str) Path to a log directory (should contain subdirectories for each run).
        :param type: (str) Type of the checkpoint ("last" or "best").
        :param trial: (int) Trial to get the checkpoint from. Defaults to last trial.
        :return: (dict) Dictionary containing information about the checkpoint.
        """
        trials = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and 'trial' in name])
        if len(trials) > 0:
            cp_path = os.path.join(path, trials[trial], "checkpoints")
        else:
            cp_path = os.path.join(path, "checkpoints")
        assert os.path.exists(cp_path), "No checkpoints directory found in {}".format(path)

        if type == "best":
            model_suffix = "best"
        else:
            model_suffix = ""

        return cls._get_latest_checkpoint(cp_path, prefix="model", suffix=model_suffix)

    @classmethod
    def get_file_path(cls, checkpoint: dict, archive_name: str, extension: str = "zip") -> str:
        """
        Returns a file according to the given checkpoint and the requested archive of that checkpoint.
        :param checkpoint: (dict) The checkpoint.
        :param archive_name: (str) The name of the archive from the given checkpoint.
        :param extension: (str) The extension of the archive.
        :return: (str) Path to the requested archive of the given checkpoint.
        """
        file_name = cls._build_filename(checkpoint, archive_name, extension=extension)
        file_path = os.path.join(checkpoint['path'], file_name)
        assert os.path.exists(file_path), "Could not find archive {} in the given checkpoint".format(file_name)
        return file_path

    @staticmethod
    def get_latest_run(path):
        runs = os.listdir(path)
        runs.sort()
        return os.path.join(path, runs[-1])  # Return latest run

    @staticmethod
    def _get_latest_checkpoint(path, prefix="", suffix=""):
        files = os.listdir(path)

        latest = datetime.fromisoformat('1970-01-01')
        counter = None
        for savepoint in files:
            datestring = os.path.splitext(savepoint)[0]
            if not (datestring.startswith(prefix) and datestring.endswith(suffix)):
                continue

            if len(prefix) > 0:
                datestring = datestring[len(prefix) + 1:]
            if len(suffix) > 0:
                datestring = datestring[:-(len(suffix) + 1)]

            step, datestring = datestring.split("_", maxsplit=1)
            # If no suffix is given the datestring may contain invalid data.
            if len(datestring) > 17:
                continue

            date = datetime.strptime(datestring, util.TIMESTAMP_FORMAT)
            if date > latest:
                latest = date
                counter = step

        return CheckpointManager._checkpoint(path, suffix, counter, latest.strftime(util.TIMESTAMP_FORMAT))

    @staticmethod
    def _checkpoint(path: str, suffix: str, counter: int, time: str) -> dict:
        """
        Creates a dictionary object describing a checkpoint.
        :param path: (str) Directory containing the checkpoint files.
        :param suffix: (str) Suffix for all the files of the checkpoint.
        :param counter: (int) Number of steps
        :param time: (str) Timestamp
        :return: (dict) Dictionary containing the checkpoint information
        """
        return {'path': path,
                'counter': counter,
                'time': time,
                'type': suffix}
