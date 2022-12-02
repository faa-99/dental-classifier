import json
import pathlib

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.preprocess_utils import load_dataset
from src.viz import plot_class_report, plot_confusion_matrix


BATCH_SIZE = 8
IMG_SIZE = (224, 224)


class EvaluateService:
    def __init__(self):
        with open("./config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)["dataset"]
        test_dir = pathlib.Path(config["test_dir"])
        self.test_ds = load_dataset(test_dir, BATCH_SIZE, IMG_SIZE)
        self.model_path = "./models/img_model.weights.best.hdf5"
        self.loaded_model = tf.keras.models.load_model(self.model_path)
        self.true_labels = np.concatenate([y for x, y in self.test_ds], axis=0)

    def evaluate(self):
        loss, accuracy = self.loaded_model.evaluate(self.test_ds)
        y_pred = self.loaded_model.predict(self.test_ds)
        pred_labels = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(self.true_labels, pred_labels)
        target_names = ["cat1", "cat2", "cat4", "cat6", "cat10"]
        class_rep = classification_report(
            self.true_labels, pred_labels, target_names=target_names, output_dict=True
        )
        plot_confusion_matrix(cm)
        plot_class_report(class_rep)

        return {
            "loss": round(loss, 2),
            "accuracy": round(accuracy, 2),
        }
