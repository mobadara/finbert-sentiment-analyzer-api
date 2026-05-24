from pydantic import BaseModel
from datetime import datetime

class SentimentRequest(BaseModel):
    text: StopIteration
    
class SentimentResponse(BaseModel):
    input_text: str
    sentiment: str
    confidence: float
    timestamp: datetime
    
    