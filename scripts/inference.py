# G:\\DL Project\\scripts\\inference.py

import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from transformers import BertTokenizer
from model import BERT_BiLSTM_Attention, SentimentClassifier
from utils import extract_embeddings, calculate_entropy

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment with confidence and entropy.",
    version="1.0.0"
)

ENCODER_PATH = "./models/encoder_model.pt"
CLASSIFIER_PATH = "./models/classifier_model.pt"

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    encoder_model = BERT_BiLSTM_Attention().to(device)
    encoder_model.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    encoder_model.eval()
    
    classifier_model = SentimentClassifier().to(device)
    classifier_model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier_model.eval()
    
    print("Models and tokenizer loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure model files are located at ./models/ in the Docker container.")
    encoder_model = None
    classifier_model = None


class TextRequest(BaseModel):
    rid: str = Field(..., example="request_123", description="Unique request identifier.")
    text: str = Field(..., example="This is a fantastic product, I really love it!", description="Text to analyze.")

class PredictionResponse(BaseModel):
    rid: str
    text: str
    label: List[str]
    sentiment_word: str
    negative_prob: float
    neutral_prob: float
    positive_prob: float
    confidence_score: float
    entropy: float



@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    """
    Receives text input and returns a detailed sentiment prediction.
    """
    if not encoder_model or not classifier_model:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    # Generate embedding for the input text
    embedding = extract_embeddings(request.text, encoder_model, tokenizer, device)

    # Get model predictions
    with torch.no_grad():
        logits = classifier_model(embedding)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] 
    # Process predictions
    prediction = np.argmax(probs)
    confidence = np.max(probs)
    entropy = calculate_entropy(probs)
    
    label_map_numeric = {0: -1, 1: 0, 2: 1}
    label_map_word = {0: "negative", 1: "neutral", 2: "positive"}

    return {
        "rid": request.rid,
        "text": request.text,
        "label": [str(label_map_numeric.get(prediction, -99))],
        "sentiment_word": label_map_word.get(prediction, "unknown"),
        "negative_prob": probs[0],
        "neutral_prob": probs[1],
        "positive_prob": probs[2],
        "confidence_score": confidence,
        "entropy": entropy
    }


@app.get("/health", status_code=200)
def health_check():
    """
    Simple endpoint to verify that the API is running and responsive.
    """
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API. Post to /predict to get a prediction."}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
