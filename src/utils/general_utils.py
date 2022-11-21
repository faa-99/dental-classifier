import shutil

from fastapi import UploadFile


def save_image(image: UploadFile, destination: str) -> str:
    with open(destination + "{}".format(image.filename), "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

        return destination + "{}".format(image.filename)
