from fastapi import APIRouter

from src.services.train_service import TrainService


router = APIRouter()

train_service = TrainService()


@router.post("/TrainCustomVGG16", tags=["model-train"])
async def train_custom_vgg16():
    history = train_service.train()
    return history
