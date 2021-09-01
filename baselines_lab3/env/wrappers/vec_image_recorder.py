from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import tile_images

from baselines_lab3.utils.recorder import GifRecorder, ImageSequenceRecorder


class VecImageRecorder(VecEnvWrapper):
    """
    Records videos from the environment in gif format.
    :param env: (VecEnv) Environment to record from.
    :param output_directory: (str) Output directory for the gifs. Individual files will be named with a timestamp
    :param record_obs: (bool) If true the recorder records observations instead of the rgb_array output of the env.
    """

    def __init__(
        self,
        env,
        output_directory,
        record_obs=False,
        format: str = "gif",
        unvec=False,
        reduction=12,
    ):
        VecEnvWrapper.__init__(self, env)
        prefix = "obs_" if record_obs else ""
        self.recorders = []
        self.reduction = reduction
        self.last = reduction
        if unvec:
            for i in range(self.num_envs):
                self.recorders.append(
                    self._create_recorder(
                        output_directory,
                        prefix="{}_{}".format(i, prefix),
                        format=format,
                    )
                )
        else:
            self.recorders.append(
                self._create_recorder(output_directory, prefix, format)
            )

        self.unvec = unvec
        self.record_obs = record_obs
        self.unvec = unvec

    def _create_recorder(self, output_dir, prefix, format):
        if format == "gif":
            recorder = GifRecorder(output_dir, prefix)
        elif format == "png":
            recorder = ImageSequenceRecorder(output_dir, prefix)
        else:
            raise ValueError("Unkown image format {}".format(format))
        return recorder

    def reset(self):
        obs = self.venv.reset()
        for recorder in self.recorders:
            recorder.reset()
        self._record(obs)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self._record(obs)
        return obs, rews, dones, infos

    def close(self):
        for recorder in self.recorders:
            recorder.close()
        VecEnvWrapper.close(self)

    def _record(self, obs):
        self.last += 1
        if self.last - self.reduction < 0:
            return
        else:
            self.last = 0

        if self.record_obs:
            if self.unvec:
                for i, recorder in enumerate(self.recorders):
                    recorder.record(obs[i])
            else:
                self.recorders[0].record(tile_images(obs))
        else:
            if self.unvec:
                images = self.venv.env_method("render", mode="rgb_array")
                for i, recorder in enumerate(self.recorders):
                    recorder.record(images[i])
            else:
                self.recorders[0].record(self.venv.render(mode="rgb_array"))
