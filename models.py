from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class ImageAnalysis(Base):
    __tablename__ = "image_analysis"

    id = Column(Integer, primary_key=True, index=True)
    image_name = Column(String, unique=True, nullable=False)
    image_date = Column(String, nullable=False)
    process_date = Column(DateTime, default=datetime.utcnow)
    water_percent = Column(Float, nullable=False)
    dry_soil_percent = Column(Float, nullable=False)
    wet_soil_percent = Column(Float, nullable=False)
    salty_muddy_percent = Column(Float, nullable=False)
    unique_key = Column(Integer, unique=True, nullable=False)
