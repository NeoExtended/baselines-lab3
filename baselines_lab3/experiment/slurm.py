import logging
import sys
import os
from pathlib import Path

from baselines_lab3.experiment.session import Session

sys.path.insert(0, os.path.dirname(os.getcwd()))

import slurminade

from baselines_lab3.utils import config_util


@slurminade.slurmify()
def run_slurm_session(config, config_path):
    config_path = Path(config_path)
    # config = config_util.read_config(config_path / "config.yml")
    logging.getLogger().setLevel(config["args"]["verbose"])

    session = Session.create_session(config, config_path)
    session.run()
    logging.info("Finished execution of config {}".format(config))
