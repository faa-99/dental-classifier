from fastapi import APIRouter
from pydantic import BaseModel

from src.services.train_service import TrainService


router = APIRouter()

train_service = TrainService()


class HyperParams(BaseModel):
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.00001
    OPTIMIZER: str = "Adam"


@router.post("/TrainCustomVGG16", tags=["model-train"])
async def train_custom_vgg16(params: HyperParams):
    print(
        f"received config {params.EPOCHS}, {params.LEARNING_RATE}, {params.OPTIMIZER}"
    )
    history = train_service.train(params.EPOCHS, params.LEARNING_RATE, params.OPTIMIZER)
    print(history)
    return {"model_history": history}
