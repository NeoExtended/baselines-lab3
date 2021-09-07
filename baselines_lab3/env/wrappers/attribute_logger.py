import json
import os
import time
from operator import attrgetter
from typing import Optional, List, Tuple

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from baselines_lab3.utils.numpy_json_encoder import NumpyJSONEncoder


class VecAttributeLogger(VecEnvWrapper):
    def __init__(
        self,
        venv: VecEnv,
        attributes: List[Tuple[str, str]],
        log_dir: Optional[str] = None,
    ):
        super(VecAttributeLogger, self).__init__(venv)
        self.reset_attributes = {a: [] for a, t in attributes if t == "reset"}
        self.done_attributes = {a: [] for a, t in attributes if t == "done"}
        self.step_attributes = {a: [] for a, t in attributes if t == "step"}

        self.log_dir = log_dir

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        for a in self.reset_attributes:
            if "." in a:
                member, attributes = a.split(".", 2)
                objects = self.venv.get_attr(member)
                self.reset_attributes[a].extend(
                    [attrgetter(attributes)(o) for o in objects]
                )
            else:
                self.reset_attributes[a].extend(self.venv.get_attr(a))
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        for a in self.step_attributes:
            if "." in a:
                member, attributes = a.split(".", 2)
                objects = self.venv.get_attr(member)
                self.step_attributes[a].extend(
                    [attrgetter(attributes)(o) for o in objects]
                )
            else:
                self.step_attributes[a].extend(self.venv.get_attr(a))
        if dones.any():
            for a in self.done_attributes:
                if "." in a:
                    member, attributes = a.split(".", 2)
                    objects = self.venv.get_attr(member)
                    self.done_attributes[a].extend(
                        [attrgetter(attributes)(o) for o, d in zip(objects, dones) if d]
                    )
                else:
                    self.done_attributes[a].extend(
                        [a for a, d in zip(self.venv.get_attr(a), dones) if d]
                    )

        return obs, rewards, dones, infos

    def close(self) -> None:
        if self.log_dir:
            if len(self.step_attributes) > 0:
                with open(
                    os.path.join(self.log_dir, f"env_attribute_log_step.json"), "w",
                ) as f:
                    json.dump(self.step_attributes, f, cls=NumpyJSONEncoder)

            if len(self.done_attributes) > 0:
                with open(
                    os.path.join(self.log_dir, f"env_attribute_log_done.json"), "w",
                ) as f:
                    json.dump(self.done_attributes, f, cls=NumpyJSONEncoder)

            if len(self.reset_attributes) > 0:
                with open(
                    os.path.join(self.log_dir, f"env_attribute_log_reset.json"), "w",
                ) as f:
                    json.dump(self.reset_attributes, f, cls=NumpyJSONEncoder)
