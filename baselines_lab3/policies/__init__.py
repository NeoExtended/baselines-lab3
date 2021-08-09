from baselines_lab3.policies.cnn_extractor import CNNExtractor, NatureCNN
from baselines_lab3.policies.custom_policy import ActorCriticCustomPolicy

from stable_baselines3.common.policies import register_policy

register_policy("ActorCriticCustomPolicy", ActorCriticCustomPolicy)
