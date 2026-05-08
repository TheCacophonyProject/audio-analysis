import logging
from pathlib import Path
import tensorflow as tf
import json


class Model:
    def __init__(self, file):
        self.model_path = Path(file)
        self.meta = None
        self.magv2 = True
        self.pre_model = False
        self.load_model()

    def load_model_meta(self):
        if self.model_path.is_file():
            meta_file = self.model_path.parent / "metadata.txt"
        else:
            meta_file = self.model_path / "metadata.txt"
        with open(meta_file, "r") as f:
            self.meta = json.load(f)

        self.pre_model = self.meta.get("pre_model", False)
        self.magv2 = self.meta.get("magv2", True)

    def load_model(self):
        logging.info("Loading model %s", self.model_path)
        if self.meta is None:
            self.load_model_meta()
        try:
            if self.magv2:
                from magtransformv2 import MagTransform
            else:
                from magtransform import MagTransform

            self.model = tf.keras.models.load_model(
                str(self.model_path),
            )

        except Exception as e:
            logging.info("Could not load model", exc_info=True)
            raise e
