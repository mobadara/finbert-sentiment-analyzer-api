from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone

from .database import Base

class InferenceLog(Base):
    __tablename__ = 'inference_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, index=True)
    sentiment_prediction = Column(String, index=True)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))