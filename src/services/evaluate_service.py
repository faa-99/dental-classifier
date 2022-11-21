import json
import pathlib

import tensorflow as tf

from src.utils.preprocess_utils import load_dataset


BATCH_SIZE = 8
IMG_SIZE = (224, 224)


class EvaluateService:
    def __init__(self):
        with open("./config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)["dataset"]
        test_dir = pathlib.Path(config["test_dir"])
        self.test_ds = load_dataset(test_dir, BATCH_SIZE, IMG_SIZE)
        self.model_path = "./models/img_model.weights.best.hdf5"

    def evaluate(self):
        loaded_model = tf.keras.models.load_model(self.model_path)
        print(loaded_model.summary())
        loss, accuracy = loaded_model.evaluate(self.test_ds)

        return {"loss": round(loss, 2), "accuracy": round(accuracy, 2)}
