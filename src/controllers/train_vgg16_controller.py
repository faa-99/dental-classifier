from fastapi import APIRouter

from src.services.train_service import TrainService


router = APIRouter()

train_service = TrainService()


@router.post("/TrainCustomVGG16", tags=["model-train"])
async def train_custom_vgg16(
    EPOCHS: int = 50,
    LEARNING_RATE: float = 0.00001,
    OPTIMIZER: str = "Adam"
):
    print(f"received config {EPOCHS}, {LEARNING_RATE}, {OPTIMIZER}")
    history = train_service.train(EPOCHS, LEARNING_RATE, OPTIMIZER)
    return history
