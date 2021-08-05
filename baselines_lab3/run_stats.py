import argparse
import os
import logging
import sys

import yaml
from pathlib import Path
from utils.tables.table_generator import TableGenerator
from utils.tensorboard import Plotter, TensorboardLogReader
from utils.tensorboard.log_reader import EvaluationLogReader


def parse_args(args):
    parser = argparse.ArgumentParser("Run script for baselines lab stats module.")

    parser.add_argument("config_file", type=str, help="Location of the lab config file. May be a list or a directory.")
    return parser.parse_args(args=args)


def make_table(config, directories):
    for i, d in enumerate(directories):
        directories[i] = Path(d)
    results = config.get('results', None)
    best = avg = drop = True
    run_id = time = std = var = cv_train = cv_test = rd_train = rd_test = False

    if results:
        best = results.get('best', True)
        avg = results.get('avg', True)
        drop = results.get('drop', True)
        run_id = results.get('run_id', False)
        time = results.get('time', False)
        std = results.get('std', False)
        var = results.get('var', False)
        cv_train = results.get('cv_train', False)
        cv_test = results.get('cv_test', False)
        rd_train = results.get('rd_train', False)
        rd_test = results.get('rd_test', False)
    format = config.get('format', 'tex')
    drop_level = config.get('drop_level', 0.05)
    max_step = config.get('max_step', None)
    output = config.get("output", "./logs")

    generator = TableGenerator.make_generator(config['headers'], files=directories, best=best, avg=avg, drop=drop,
                                              run_id=run_id, time=time, std=std, var=var, cv_train=cv_train, cv_test=cv_test,
                                              rd_test=rd_test, rd_train=rd_train)
    generator.make_table(output, format=format, drop_level=drop_level, max_step=max_step)


def make_figure(config, directories):
    file_format = config.get("format", "pdf")
    output_dir = config.get("output", "./logs")
    source = config.get("source", "tensorboard")

    plot_avg_only = config.get("plot_avg_only", False)
    smoothing = config.get("smoothing", 0.9)
    alias = config.get("alias", None)
    max_step = config.get("max_step", None)
    trial = config.get("trial", None)
    step_type = config.get("step_type", "step")

    if step_type == "step":
        x_label = "Steps"
    else:
        x_label = "Time (in minutes)"

    if source == "tensorboard":
        tags = config.get("tags", ["episode_length/ep_length_mean"])
        names = config.get("names", ["Episode Length"])
        if len(tags) != len(names):
            raise ValueError("There must be a name for each tag and vice versa!")

        reader = TensorboardLogReader(directories)

    elif source == "evaluation":
        tags = config.get("tags", ["distance", "n_particles", "max_distance"])
        names = config.get("names", ["Total Distance", "Unique Particles", "Maximum Distance"])
        reader = EvaluationLogReader(directories)
    else:
        raise ValueError("Unknown source {}".format(source))

    plot = Plotter(output_dir, file_format=file_format)
    plot.from_reader(reader, tags=tags, names=names, plot_avg_only=plot_avg_only, smoothing=smoothing, alias=alias,
                     max_step=max_step, trial=trial, step_type=step_type, x_label=x_label)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    logging.getLogger().setLevel(logging.INFO)
    config_file = Path(args.config_file)
    tasks = []

    if config_file.is_dir():
        tasks.extend(config_file.glob("**/*.yml"))
    else:
        tasks = [config_file]

    for task in tasks:
        file = task.open("r")
        config = yaml.load(file)
        file.close()

        directories = []
        for dir in config["runs"]:
            filelist = os.listdir(dir)
            if "config.yml" in filelist:  # Directory is a run dir
                directories.append(dir)
            else:  # Directory contains multiple run dirs.
                directories.extend([f.path for f in os.scandir(dir) if f.is_dir()])

        for job in config["jobs"]:
            type = job["type"]
            if type == "figure":
                make_figure(job, directories)
            elif type == "table":
                make_table(job, directories)


if __name__ == "__main__":
    main()

