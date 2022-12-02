from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import Adadelta, Adamax, RMSprop
from tensorflow.keras.utils import image_dataset_from_directory


def load_dataset(dataset_dir, batch_size, image_size):
    return image_dataset_from_directory(
        dataset_dir, shuffle=False, image_size=image_size, batch_size=batch_size
    )


def get_augmentation_layers():
    return Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ]
    )


def get_early_stopping_callback():
    """
    Early stopping to prevent over training and to ensure decreasing validation loss

    :return:
    """
    return EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, mode="min"
    )


def get_checkpoint_callback():
    """
    Saves Keras model after each epoch
    :return:
    """
    return ModelCheckpoint(
        filepath="./models/latest_model_weights.hdf5",
        verbose=1,
        save_best_only=True,
    )


def get_optimizer(OPTIMIZER, LEARNING_RATE):
    match OPTIMIZER:
        case "Adam":
            return Adam(learning_rate=LEARNING_RATE)
        case "AdaMax":
            return Adamax(learning_rate=LEARNING_RATE)
        case "Adadelta":
            return Adadelta(learning_rate=LEARNING_RATE)
        case "RMSprop":
            return RMSprop(learning_rate=LEARNING_RATE)
