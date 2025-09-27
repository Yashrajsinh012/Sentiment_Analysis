import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from model import BERT_BiLSTM_Attention, SentimentClassifier
from scripts.utils import load_and_combine_data, extract_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pairwise_distance(embeddings):
    dot = torch.matmul(embeddings, embeddings.T)
    norm = torch.diagonal(dot)
    return torch.clamp(norm.unsqueeze(1) - 2 * dot + norm.unsqueeze(0), min=0.0)

def batch_hard_triplet_loss(embeddings, labels, margin=0.5):
    dists = pairwise_distance(embeddings)
    labels = labels.unsqueeze(1)
    mask_pos = (labels == labels.T).float()
    mask_neg = (labels != labels.T).float()
    hardest_pos = (dists * mask_pos).max(dim=1)[0]
    hardest_neg = (dists + dists.max().item() * (1 - mask_neg)).min(dim=1)[0]
    return torch.relu(hardest_pos - hardest_neg + margin).mean()

class TripletSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.tokenizer, self.max_len = tokenizer, max_len
        self.class_to_indices = {
            label: df[df['sentiment'] == label].reset_index(drop=True) for label in df['sentiment'].unique()
        }
        # Precompute lengths for __len__
        self.length = min(len(v) for v in self.class_to_indices.values())

    def __len__(self):
        return self.length

    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")

    def __getitem__(self, idx):
        anchor_sentiment = random.choice(list(self.class_to_indices.keys()))
        anchor_row = self.class_to_indices[anchor_sentiment].iloc[idx]
        anchor_text = anchor_row['summary']

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.class_to_indices[anchor_sentiment].index.tolist())
        positive_text = self.class_to_indices[anchor_sentiment].loc[pos_idx, 'summary']

        negative_classes = [label for label in self.class_to_indices if label != anchor_sentiment]
        neg_class = random.choice(negative_classes)
        neg_idx = random.choice(self.class_to_indices[neg_class].index.tolist())
        negative_text = self.class_to_indices[neg_class].loc[neg_idx, 'summary']

        anchor = self.tokenize(anchor_text)
        pos = self.tokenize(positive_text)
        neg = self.tokenize(negative_text)

        return {
            'anchor_input_ids': anchor['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor['attention_mask'].squeeze(0),
            'pos_input_ids': pos['input_ids'].squeeze(0),
            'pos_attention_mask': pos['attention_mask'].squeeze(0),
            'neg_input_ids': neg['input_ids'].squeeze(0),
            'neg_attention_mask': neg['attention_mask'].squeeze(0)
        }

def train_triplet_encoder(model, df, tokenizer, batch_size=32, patience=5, max_epochs=10):
    model.train()
    loader = DataLoader(TripletSentimentDataset(df, tokenizer), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_train_loss = float('inf')
    patience_counter = 0
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        total_loss = 0
        model.train()
        for batch in tqdm(loader, desc=f"Triplet Epoch {epoch}"):
            input_ids = torch.cat([batch['anchor_input_ids'], batch['pos_input_ids'], batch['neg_input_ids']], dim=0).to(device)
            attention_mask = torch.cat([batch['anchor_attention_mask'], batch['pos_attention_mask'], batch['neg_attention_mask']], dim=0).to(device)
            labels = torch.cat([
                torch.zeros(len(batch['anchor_input_ids'])),
                torch.ones(len(batch['pos_input_ids'])),
                torch.full((len(batch['neg_input_ids']),), 2)
            ]).long().to(device)

            embeddings = model(input_ids, attention_mask)
            loss = batch_hard_triplet_loss(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")

        if avg_loss < best_train_loss - 1e-3:
            best_train_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no significant improvement)")
                break

def train_classifier(encoder, classifier, df, tokenizer, batch_size=64, patience=5, delta=0.001):
    classifier.train()
    embeddings, labels = extract_embeddings(df['summary'].tolist(), encoder, tokenizer, device=device)
    labels = torch.tensor([label + 1 for label in df['sentiment']], dtype=torch.long).to(device)
    dataset = DataLoader(TensorDataset(torch.tensor(embeddings, dtype=torch.float32), labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_loss = float('inf')
    epochs_without_improvement = 0

    while epochs_without_improvement < patience:
        total_loss = 0
        for X, y in dataset:
            X, y = X.to(device), y.to(device)
            logits = classifier(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataset)
        print(f"Avg Loss: {avg_loss:.4f}")

        if best_loss - avg_loss > delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"Epochs without improvement: {epochs_without_improvement}")

    print("Early stopping triggered!")

if __name__ == "__main__":
    annotated_jsonl_path = "/teamspace/studios/this_studio/Final_pipeline/manually_annotated/iter_07.jsonl"
    initial_labeled_jsonl_path = "/teamspace/studios/this_studio/Final_pipeline/updated_train.jsonl"

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    df_combined = load_and_combine_data(annotated_jsonl_path, initial_labeled_jsonl_path)
    encoder = BERT_BiLSTM_Attention().to(device)
    classifier = SentimentClassifier().to(device)

    train_triplet_encoder(encoder, df_combined, tokenizer)
    train_classifier(encoder, classifier, df_combined, tokenizer)

    torch.save(classifier.state_dict(), "/teamspace/studios/this_studio/Final_pipeline/models/total_finetuning/classifier/classifier_model5.pt")
    torch.save(encoder.state_dict(), "/teamspace/studios/this_studio/Final_pipeline/models/total_finetuning/encoder/encoder_model5.pt")
    print("Fine-tuned models saved!")
