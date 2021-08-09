from typing import List, Tuple, Union, Type

import gym
import torch
import torch.nn as nn

from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        arch: List[Union[Tuple[str, int, int], Tuple[str, int, int, int]]],
        features_dim: int = 512,
        activation: Type[nn.Module] = nn.LeakyReLU,
    ):
        super(CNNExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        self._build_cnn(observation_space, arch, activation)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), activation())

    def _build_cnn(
        self,
        observation_space: gym.spaces.Box,
        arch: List[Union[Tuple[str, int, int], Tuple[str, int, int, int]]],
        activation: nn.Module = nn.LeakyReLU,
    ):
        current_input = observation_space.shape[0]
        components = []
        for part in arch:
            if part[0] == "pool":
                _, kernel_size, stride = part
                components.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
            elif part[0] == "conv":
                _, n_filters, kernel_size, stride = part
                components.append(
                    nn.Conv2d(
                        current_input,
                        n_filters,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=0,
                    )
                )
                components.append(activation())
                current_input = n_filters
        components.append(nn.Flatten())
        self.cnn = nn.Sequential(*components)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear.forward(self.cnn.forward(observations))


class NatureCNN(CNNExtractor):
    """
    CNN from DQN nature paper with variable activation function:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        activation: nn.Module = nn.LeakyReLU,
    ):
        arch = [("conv", 32, 8, 4), ("conv", 64, 4, 2), ("conv", 64, 3, 1)]
        super(NatureCNN, self).__init__(
            observation_space, arch, features_dim, activation
        )
