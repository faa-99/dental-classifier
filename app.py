import json

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.controllers.classify_controller import router as classify_router
from src.controllers.dataset_controller import router as dataset_router
from src.controllers.evaluate_controller import router as evaluate_router
from src.controllers.health_controller import router as health_router
from src.controllers.train_vgg16_controller import router as train_vgg16_router


with open("config.json", "r", encoding="utf-8") as config_file:
    config = json.load(config_file)

app = FastAPI(title=config["dental-classifier"]["api_name"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routes_prefix = config["dental-classifier"]["controllers"]["prefix"]

app.include_router(health_router)

app.include_router(dataset_router, prefix=routes_prefix)

app.include_router(evaluate_router, prefix=routes_prefix)

app.include_router(train_vgg16_router, prefix=routes_prefix)

app.include_router(classify_router, prefix=routes_prefix)
