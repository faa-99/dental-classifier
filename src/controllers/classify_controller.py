import json

from fastapi import APIRouter, File, UploadFile

from src.services.classify_service import ClassifyService
from src.utils.general_utils import save_image


router = APIRouter()

classify_service = ClassifyService()

with open("./config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)["classify"]
path_to_save = config["save_image_path"]


@router.post("/ClassifyImage", tags=["classify"])
async def classify_image(image: UploadFile = File(description="Image to classify")):
    path_to_uploaded_image = save_image(image=image, destination=path_to_save)
    print(f"path to uploaded image is {path_to_uploaded_image}")
    label, confidence = classify_service.classify_image(
        source_image=path_to_uploaded_image
    )
    return {"label": label, "confidence": confidence}
