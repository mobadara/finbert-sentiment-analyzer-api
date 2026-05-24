from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from .database import get_db
from .models import InferenceLog
from .schemas import SentimentRequest, SentimentResponse
from . import ml_model

app = FastAPI(
  title='FinBERT Sentiment Analyzer API',
  description='An API for analyzing the sentiment of financial news articles using FinBERT.',
  version='1.0.0'
)

@app.post('/predict', response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest, db: Session = Depends(get_db)):
  try:
    prediction_result = ml_model.predict(request.text)
    log_entry = InferenceLog(
      input_text=request.text,
      sentiment_prediction=prediction_result['sentiment'],
      confidence_score=prediction_result['confidence'],
      timestamp=datetime.now(timezone.utc)
    )
    
    db.add(log_entry)
    db.commit()
    db.close()
    
    return SentimentResponse(
      input_text=request.text,
      sentiment=prediction_result['sentiment'],
      confidence=prediction_result['confidence'],
      timestamp=log_entry.timestamp
    )
  except Exception as e:
    raise HttpException(status_code=500, detail=str(e))
  
  
@app.get('/logs')
def get_inference_logs(db: Session = Depends(get_db)):
  logs = db.query(InferenceLog).order_by(InferenceLog.timestamp.desc()).all()
  return logs


@app.get('/health')
def health_check():
  return {"status": "ok", "timestamp": datetime.now(timezone.utc)}