import json
import pathlib

import tensorflow as tf
from fastapi import APIRouter


router = APIRouter()


@router.get("/GetImageNumber", tags=["dataset"])
async def get_image_number():
    with open("./config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)["dataset"]
    data_dir = pathlib.Path(config["original_dataset_directory"])
    image_count = len(list(data_dir.glob("*/*.jpg")))
    return {"images": image_count}


@router.get("/GetClasses", tags=["dataset"])
async def get_classes():
    with open("./config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)["dataset"]
    data_dir = pathlib.Path(config["original_dataset_directory"])
    dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    class_names = dataset.class_names

    return class_names
