from os.path import exists

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


INPUT_SHAPE = (224, 224, 3)
MODEL_PATH = "./models/vgg_base_model.h5"


def get_vgg16_model():
    if exists(MODEL_PATH):
        print("Model exists. Loading it...")
        vgg16_model = tf.keras.models.load_model(MODEL_PATH)
    else:
        vgg16_model = VGG16(
            weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
        )
        for layer in vgg16_model.layers:
            layer.trainable = False

        print(vgg16_model.summary())
        vgg16_model.save(MODEL_PATH)
    return vgg16_model
