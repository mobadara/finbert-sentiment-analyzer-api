from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from .database import get_db, Base, engine
from .models import InferenceLog
from .schemas import SentimentRequest, SentimentResponse
from . import ml_model

Base.metadata.create_all(bind=engine)  # Ensure tables are created at startup

app = FastAPI(
  title='FinBERT Sentiment Analyzer API',
  description='An API for analyzing the sentiment of financial news articles using FinBERT.',
  version='1.0.0',
  docs_url='/docs',
  redoc_url='/'
)


origins = [
    'http://localhost:3000',                  
    'https://portfolio-frontend-livid.vercel.app',
    '*'                                       
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all standard methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all standard headers
)

@app.post('/predict', response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest, db: Session = Depends(get_db)):
  try:
    request_data = request.model_dump() 
    prediction_result = ml_model.predict(request_data["text"])
    current_time = datetime.now(timezone.utc)
    log_entry = InferenceLog(
      input_text=request_data['text'],
      sentiment_prediction=prediction_result['sentiment'],
      confidence_score=prediction_result['confidence'],
      timestamp=current_time
    )
    
    db.add(log_entry)
    db.commit()
    
    return SentimentResponse(
      input_text=request_data["text"],
      sentiment=prediction_result['sentiment'],
      confidence=prediction_result['confidence'],
      timestamp=log_entry.timestamp
    )
  except Exception as e:
    db.rollback()  # Rollback in case of any error during database operations
    raise HTTPException(status_code=500, detail=str(e))


@app.get('/logs')
def get_inference_logs(db: Session = Depends(get_db)):
  logs = db.query(InferenceLog).order_by(InferenceLog.timestamp.desc()).all()
  return logs


@app.get('/health')
def health_check():
  return {"status": "ok", "timestamp": datetime.now(timezone.utc)}