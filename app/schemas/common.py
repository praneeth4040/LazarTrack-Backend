from typing import Any, Dict, List
from pydantic import BaseModel, Field


class MatchPlayerStats(BaseModel):
    inGameName: str
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage: int = 0
    actualDamage: int = 0
    knockedDown: int = 0
    heal: int = 0
    helpUp: int = 0
    revival: int = 0
    headShotRate: float = 0.0


class ExtractionResponse(BaseModel):
    pipeline_id: str
    pipeline_name: str
    confidence: float = 1.0
    data: Dict[str, Any]
    raw_texts: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    easyocr_url: str
    pipelines_available: List[str]
