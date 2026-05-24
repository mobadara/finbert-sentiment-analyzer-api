import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Target your specific Hugging Face repository
HF_MODEL_REPO = 'mobadara/finbert-finetuned'

logging.info(f'Initializing NLP pipeline from {HF_MODEL_REPO}...')

# Load tokenizer and model weights
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
model.eval()  # Lock the model in evaluation mode for inference

# Map model output indices to our target classes
LABEL_MAPPING = {0: 'Negative', 1: 'Neutral', 2: "Positive"}

def predict(text: str) -> dict:
    """
    Takes raw text, tokenizes it, runs it through FinBERT, 
    and returns the predicted sentiment and confidence score.
    
    Args:
        text (str): The input text to analyze.
        
    Returns:
        dict: A dictionary containing the predicted sentiment and confidence score.
        
    Throws:
        ValueError: If the input text is empty or None.
    """
    if not text:
        raise ValueError("Input text cannot be empty or None.")

    # Tokenize the incoming text
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Perform inference without tracking gradients (saves memory/time)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Convert raw logits to probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Extract the highest probability and its index
        confidence_score, predicted_class_idx = torch.max(probabilities, dim=1)
        
        sentiment_label = LABEL_MAPPING[predicted_class_idx.item()]
        confidence_float = round(confidence_score.item(), 4)
        
    return {
        'sentiment': sentiment_label,
        'confidence': confidence_float
    }