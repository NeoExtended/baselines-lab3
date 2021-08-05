from stable_baselines.common.policies import register_policy

from baselines_lab3.policies.cnn_policy import SimpleMazeCnnPolicy, GeneralCnnPolicy
from baselines_lab3.policies.rnd_policy import RndPolicy
from baselines_lab3.policies.deepq import FeedForwardPolicy

register_policy('RndPolicy', RndPolicy)
register_policy('SimpleMazeCnnPolicy', SimpleMazeCnnPolicy)
register_policy('GeneralCnnPolicy', GeneralCnnPolicy)
register_policy('GeneralDqnPolicy', FeedForwardPolicy)