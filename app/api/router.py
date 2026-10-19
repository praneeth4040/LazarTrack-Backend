from fastapi import APIRouter
from app.api.v1 import health, extract, visualize, battle_log, zone_tracker

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(extract.router)
api_router.include_router(visualize.router)
api_router.include_router(battle_log.router)
api_router.include_router(zone_tracker.router)
