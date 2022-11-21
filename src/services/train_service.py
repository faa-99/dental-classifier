import json
import pathlib

from src.model.custom_vgg16 import create_model
from src.utils.preprocess_utils import (
    get_checkpoint_callback,
    get_early_stopping_callback,
    load_dataset,
)
from src.viz import plot_accuracy_and_loss


BATCH_SIZE = 8
EPOCHS = 60
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)


class TrainService:
    def __init__(self):
        with open("./config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)["dataset"]

        train_dir = pathlib.Path(config["train_dir"])
        val_dir = pathlib.Path(config["val_dir"])

        self.train_ds = load_dataset(train_dir, BATCH_SIZE, IMG_SIZE)
        self.val_ds = load_dataset(val_dir, BATCH_SIZE, IMG_SIZE)

    def train(self):
        check_point_callback = get_checkpoint_callback()
        early_stop_callback = get_early_stopping_callback()
        model = create_model()

        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=EPOCHS,
            callbacks=[check_point_callback, early_stop_callback],
            verbose=1,
        )
        plot_accuracy_and_loss(history.history)
        print(model.summary())
        return history.history
