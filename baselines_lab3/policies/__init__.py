from baselines_lab3.policies.cnn_extractor import CNNExtractor, NatureCNN
from baselines_lab3.policies.combined_extractor import CombinedExtractor
from baselines_lab3.policies.custom_policy import ActorCriticCustomPolicy

from stable_baselines3.ppo import PPO
from stable_baselines3.a2c import A2C

PPO.policy_aliases["ActorCriticCustomPolicy"] = ActorCriticCustomPolicy
A2C.policy_aliases["ActorCriticCustomPolicy"] = ActorCriticCustomPolicy
