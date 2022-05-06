import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import simple_slurm
import slurminade

from baselines_lab3.experiment.slurm import run_slurm_session
from baselines_lab3.experiment.session import Session
from baselines_lab3.utils import send_email, config_util, util
from slurminade.conf import _get_conf


class Scheduler:
    """
    The scheduler class manages the execution of the lab experiments.
    If multiple experiments have been specified they are run one by one.
    :param configs: (list) List of lab configs to execute.
    :param args: (namespace) Lab arguments.
    """

    def __init__(self, configs: List[Dict], args: Namespace):
        self.configs = configs
        self.args = args

    def _maybe_send_mail(self, title: str, text: str):
        if self.args.mail:
            send_email(self.args.mail, title, text)

    def _create_log_dir(self, config: Dict):
        log_location = config["meta"].get("log_dir", None)
        log_dir = util.create_log_directory(log_location)
        config["meta"]["timestamp"] = util.get_timestamp()
        config_util.save_config(config, log_dir / "config.yml")
        return log_dir

    def _schedule_config(self, config: Dict, log_dir: Path):
        n_trials = config["meta"].get("n_trials", 1)

        # Do not run multiple trials in search or enjoy mode
        lab_mode = config["args"].get("lab_mode")
        if not lab_mode == "train":
            n_trials = 1

        if n_trials == 1:
            # Do not create an extra log directory for the trial if there is only a single trial.
            config["meta"]["generated_seed"] = config_util.seed_from_config(config)
            config_util.save_config(config, log_dir / "config.yml")
            self._schedule_trial(config, log_dir)
        else:
            for trial in range(n_trials):
                config["meta"]["generated_seed"] = config_util.seed_from_config(
                    config, increment=trial
                )
                trial_dir = log_dir / f"trial_{trial}"
                trial_dir.mkdir()
                util.set_log_directory(trial_dir)
                config_util.save_config(config, trial_dir / "config.yml")

                self._schedule_trial(config, trial_dir)

    def _schedule_local(self, config: Dict, log_dir: Path):
        session = Session.create_session(config, log_dir)
        session.run()
        logging.info("Finished execution of config {}".format(config))

    def _schedule_distributed(self, config: Dict, log_dir: Path):
        if config["meta"].get("slurm", False):
            slurminade.update_default_configuration(**config["meta"].get("slurm"))

        # TODO: Remove this weird patch once slurmify allows live argument updates.
        conf = _get_conf({})
        run_slurm_session.slurm = simple_slurm.Slurm(**conf)
        run_slurm_session.distribute(str(log_dir))
        logging.info("Scheduled configuration {}".format(config))

    def _schedule_trial(self, config: Dict, log_dir: Path):
        if config["args"]["distributed"]:
            self._schedule_distributed(config, log_dir)
        else:
            self._schedule_local(config, log_dir)

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
                self._schedule_config(config, log_dir)
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
