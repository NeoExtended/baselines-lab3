import logging
from pathlib import Path
from typing import Dict

from baselines_lab3.experiment.session import Session
from baselines_lab3.utils import send_email, config_util, util


class Scheduler:
    """
    The scheduler class manages the execution of the lab experiments.
    If multiple experiments have been specified they are run one by one.
    :param configs: (list) List of lab configs to execute.
    :param args: (namespace) Lab arguments.
    """

    def __init__(self, configs, args):
        self.configs = configs
        self.args = args

    def _maybe_send_mail(self, title, text):
        if self.args.mail:
            send_email(self.args.mail, title, text)

    def _create_log_dir(self, config):
        log_location = config["meta"].get("log_dir", None)
        log_dir = util.create_log_directory(log_location)
        config["meta"]["timestamp"] = util.get_timestamp()
        config_util.save_config(config, log_dir / "config.yml")
        return log_dir

    def schedule_config(self, config: Dict, log_dir: Path):
        n_trials = config["meta"].get("n_trials", 1)

        # Do not run multiple trials in search or enjoy mode
        lab_mode = config["args"].get("lab_mode")
        if not lab_mode == "train":
            n_trials = 1

        if n_trials == 1:
            # Do not create an extra log directory for the trial if there is only a single trial.
            config["meta"]["generated_seed"] = config_util.seed_from_config(config)
            config_util.save_config(config, log_dir / "config.yml")
            self.schedule_trial(config, log_dir)
        else:
            for trial in range(n_trials):
                config["meta"]["generated_seed"] = config_util.seed_from_config(
                    config, increment=trial
                )
                trial_dir = log_dir / f"trial_{trial}"
                trial_dir.mkdir()
                util.set_log_directory(trial_dir)
                config_util.save_config(config, trial_dir / "config.yml")

                self.schedule_trial(config, trial_dir)

    def schedule_trial(self, config, log_dir: Path):
        session = Session.create_session(config, log_dir)
        session.run()
        logging.info("Finished execution of config {}".format(config))

    def run(self):
        for config in self.configs:
            try:
                if config["args"].get("lab_mode") == "search" and config["search"].get(
                    "resume", False
                ):
                    # Do not create a new log directory when resuming a hyperparameter search
                    log_dir = Path(config["search"]["resume"])
                else:
                    log_dir = self._create_log_dir(config)
                self.schedule_config(config, log_dir)
            except Exception as err:
                logging.error(
                    "An exception {} occurred when executing config {} with args {}".format(
                        err, config, self.args
                    )
                )
                if not self.args.ignore_errors:
                    self._maybe_send_mail(
                        "Run Failed",
                        "Training for config {} with args {} failed.".format(
                            config, self.args
                        ),
                    )
                    raise err

            self._maybe_send_mail(
                "Finished Training",
                "Finished training for config {} with args {}.".format(
                    config, self.args
                ),
            )
