import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from src.model.vgg16 import get_vgg16_model
from src.utils.preprocess_utils import get_augmentation_layers, get_optimizer


NUM_CLASSES = 5
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)


def create_model(LEARNING_RATE: float, OPTIMIZER: str):
    data_augmentation = get_augmentation_layers()

    base_model = get_vgg16_model()

    optimizer = get_optimizer(OPTIMIZER, LEARNING_RATE)

    model = tf.keras.Sequential(
        [
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.Conv2D(
                filters=512, kernel_size=3, activation="relu", padding="same"
            ),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, activation="relu", padding="same"
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=4),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CLASSES),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
