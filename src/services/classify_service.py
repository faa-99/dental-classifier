import json
import pathlib
from typing import Tuple

import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)


class ClassifyService:
    def __init__(self):
        print("Initialized Classify Service.")
        with open("./config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        data_dir = pathlib.Path(config["dataset"]["original_dataset_directory"])
        dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
        self.class_names = dataset.class_names
        self.model_path = config["model"]["path"]
        self.loaded_model = tf.keras.models.load_model(self.model_path)
        print(self.loaded_model.summary())

    def classify_image(self, source_image) -> Tuple[str, float]:
        image = tf.keras.utils.load_img(source_image, target_size=IMG_SIZE)
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)  # Create a batch
        predictions = self.loaded_model.predict(image_array)
        score = tf.nn.softmax(predictions[0])
        label = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                label, confidence
            )
        )
        return label, confidence
