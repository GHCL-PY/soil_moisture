from pydantic import BaseModel
from datetime import datetime

class AnalysisResult(BaseModel):
    filename: str
    water_percentage: float
    dry_percentage: float
    wet_percentage: float
    salty_muddy_percentage: float
    timestamp: datetime
