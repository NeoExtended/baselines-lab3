import logging
import os
from abc import ABC, abstractmethod

import imageio
import numpy as np

from baselines_lab3.utils import get_timestamp


class Recorder(ABC):
    def __init__(self, path: str, name_prefix: str = ""):
        self.path = path
        self.name_prefix = name_prefix
        self.images = []

    def record(self, image):
        self.images.append(image)

    def reset(self):
        self._save_images()
        self.images = []

    def close(self):
        self.reset()

    @abstractmethod
    def _save_images(self):
        pass


class GifRecorder(Recorder):
    def _save_images(self):
        if len(self.images) > 0:
            gif_path = os.path.join(
                self.path, "{}{}.gif".format(self.name_prefix, get_timestamp())
            )
            logging.info("Saving gif to {}".format(gif_path))
            imageio.mimsave(gif_path, [np.array(img) for img in self.images], fps=30)


class ImageSequenceRecorder(Recorder):
    def _save_images(self):
        if len(self.images) > 0:
            image_path = os.path.join(
                self.path, "{}_{}".format(self.name_prefix, get_timestamp())
            )
            logging.info("Saving images to {}".format(image_path))
            os.makedirs(image_path, exist_ok=True)
            for i, image in enumerate(self.images):
                imageio.imwrite(
                    os.path.join(image_path, "{}.png".format(str(i))), image
                )
