import os
import sys

# Allow absolute imports even if project is not installed
sys.path.insert(0, os.path.dirname(os.getcwd()))

import argparse
import logging

from baselines_lab3.utils import config_util
from baselines_lab3.experiment import Scheduler


def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab.")
    subparsers = parser.add_subparsers(
        help="Argument defining the lab mode.", dest="lab_mode", required=True
    )

    enjoy_parser = subparsers.add_parser("enjoy")
    enjoy_parser.add_argument(
        "--type",
        help="Checkpoint type to load",
        choices=["best", "last"],
        default="best",
    )
    enjoy_parser.add_argument(
        "--checkpoint-path",
        help="Path to a directory containing a model checkpoint (defaults to config log dir)",
        default="",
    )
    enjoy_parser.add_argument(
        "--video", help="Create a video file in enjoy mode", action="store_true"
    )
    enjoy_parser.add_argument(
        "--obs-video",
        help="Create a video file capturing the observations (only works if the env outputs image-like obs)",
        action="store_true",
    )
    enjoy_parser.add_argument(
        "--stochastic",
        help="Execute the neural network in stochastic instead of deterministic mode.",
        action="store_true",
    )
    enjoy_parser.add_argument(
        "--evaluate",
        help="Activates the model evaluation over at least x given episodes and saves the result to the model dir. "
        "(Use with --strict option for more accurate evaluation!)",
        type=int,
        default=None,
    )
    enjoy_parser.add_argument(
        "--strict",
        help="Sets the number of environments to 1. Results in more accurate but far slower evaluation.",
        action="store_true",
    )
    enjoy_parser.add_argument(
        "--trial",
        type=int,
        help="Trial to load for enjoy mode (defaults to last trial).",
        default=-1,
    )
    enjoy_parser.add_argument(
        "--plot", help="Weather or not to plot tensorboard data.", action="store_true"
    )
    enjoy_parser.add_argument(
        "--random-agent",
        help="Weather or not to use a random agent",
        action="store_true",
    )

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last log directory. Uses the last agent checkpoint.",
    )
    train_parser.add_argument(
        "--trial",
        type=int,
        help="Trial to load when resuming training (defaults to last trial from previous run).",
        default=-1,
    )
    train_parser.add_argument(
        "--video",
        action="store_true",
        help="Stores a video for each episode of training (Warning: May consume a lot of memory)",
    )
    train_parser.add_argument(
        "--video-format",
        choices=["gif", "png"],
        default="gif",
        help="Select the video file format.",
    )

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last log directory. Uses the last agent checkpoint.",
    )
    search_parser.add_argument(
        "--plot",
        help="Weather or not to plot the distribution of choosen hyperparameters",
        action="store_true",
    )

    parser.add_argument(
        "config_file",
        type=str,
        nargs="+",
        help="Location of the lab config file. May be a list or a directory.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=logging.INFO,
        help="Verbosity level - corresponds to python logging levels",
    )
    parser.add_argument(
        "--mail",
        type=str,
        default=None,
        help="Set your mail address to be informed when training is finished (requires mailx)",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Weather or not to suppress errors when executing multiple configs.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="If set, sessions will be distributed using slurm.",
    )
    return parser.parse_args(args=args)


def main(args=None):
    # TODO: Multi-Level obs videos: Provide obs videos after each? wrapper.

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(args.verbose)
    configs = config_util.parse_config_args(args.config_file, args)

    s = Scheduler(configs, args)
    s.run()


if __name__ == "__main__":
    main()
