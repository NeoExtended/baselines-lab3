[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Baselines Lab 3
Zero-code configuration file based experimentation environment for reinforcement learning. 
Built on top of the wonderful implementations of [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

Baselines-Lab 3 makes it easy to experiment with RL algorithms and supports rapidly changing configurations without the need to rewrite any code. 
Everything you need to do is write simple configuration files (see examples below).

# Main Features
- Zero-code reinfocement learning with configuration files
- Expanded (tensorboard) logging capabilities
- Automated hyperparameter optimization (with [optuna](https://github.com/optuna/optuna))
- Automated model evaluation, saving and loading

# Installation
Currently there is no pip-package available. 
To install please clone this directory. Then install with pip using
```
pip install -e baselines-lab3
```


# Example Configurations
There are a few example configuration files available in the [config](config) directory. 

Each configuration file is divided into up to four sub-sections: 
The `algorithm`, `env`, `meta` and `search` sections. 

TODO


# Running Experiments
The lab is started using the script ``run_lab.py``. 
Use one of the three lab-modes to perform different actions:
1. Start a new training session, by running ``python run_lab.py train path/to/config-file.yml``
2. Observe what the network is doing, or start network evaluation using ``python run_lab.py enjoy path/to/config-file.yml``
3. Automatically optimize hyperparameters, by running ``python run_lab.py search path/to/config-file.py``

# Additional CLI Arguments
TODO
# Configuration Keywords
TODO