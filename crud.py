from sqlalchemy.orm import Session
from models import ImageAnalysis

def save_analysis_result(db: Session, filename: str, analysis: dict):
    db_entry = ImageAnalysis(
        filename=filename,
        water_percentage=analysis["water"],
        dry_percentage=analysis["dry"],
        wet_percentage=analysis["wet"],
        salty_muddy_percentage=analysis["salty_muddy"]
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
