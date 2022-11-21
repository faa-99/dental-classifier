from fastapi import APIRouter

from src.services.evaluate_service import EvaluateService


router = APIRouter()

evaluate_service = EvaluateService()


@router.get("/EvaluateModel", tags=["model-evaluate"])
async def evaluate():
    response = evaluate_service.evaluate()
    return response
