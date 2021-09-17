from typing import Union, Type, Dict, Any

import torch
import torch.nn as nn
import gym
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict

from baselines_lab3 import utils
from baselines_lab3.policies import NatureCNN


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        image_extractor_class: Union[Type[BaseFeaturesExtractor], str] = NatureCNN,
        cnn_output_dim: int = 512,
        image_extractor_kwargs: Dict[str, Any] = None,
    ):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        if isinstance(image_extractor_class, str):
            image_extractor_class = utils.load_class_from_module(image_extractor_class)

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = image_extractor_class(
                    subspace, features_dim=cnn_output_dim, **image_extractor_kwargs
                )
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
