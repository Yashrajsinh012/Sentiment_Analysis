import json
import logging
import torch
import numpy as np
import pandas as pd

def extract_embeddings(text: str, encoder, tokenizer, device):
    """
    Extracts embedding for a single text input.
    """
    encoder.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        embedding = encoder(input_ids, attention_mask)
        return embedding
    

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)


def calculate_entropy(probs):
    probs = np.asarray(probs)
    ent = -np.sum(probs * np.log2(probs + 1e-9), axis=-1)  # use last axis
    return ent

def load_data(path):
    df = pd.read_parquet(path)
    if 'text' not in df.columns:
        raise ValueError(f"'text' column is required")
    df['summary'] = df['text'].astype(str).str.strip()
    return df


