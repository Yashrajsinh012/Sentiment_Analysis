import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import BertTokenizer
from model import BERT_BiLSTM_Attention, SentimentClassifier
from scripts.utils import load_jsonl, extract_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    test_data_path = "/teamspace/studios/this_studio/Final_pipeline/test_data.jsonl"
    output_log_path = "/teamspace/studios/this_studio/Final_pipeline/test/test_06.log"
    confusion_matrix_path = "/teamspace/studios/this_studio/Final_pipeline/test/confusion_matrix_06.png"
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    logging.basicConfig(filename=output_log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting inference on test data...")

    test_data = load_jsonl(test_data_path)
    test_texts = [item['text'] for item in test_data]
    label_map = {"-1": 0, "0": 1, "1": 2}
    test_labels = [label_map[item['label']] for item in test_data]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoder = BERT_BiLSTM_Attention().to(device)
    classifier = SentimentClassifier().to(device)

    encoder.load_state_dict(torch.load("/teamspace/studios/this_studio/Final_pipeline/models/total_finetuning/encoder/encoder_model3.pt", map_location=device))
    classifier.load_state_dict(torch.load("/teamspace/studios/this_studio/Final_pipeline/models/total_finetuning/classifier/classifier_model3.pt", map_location=device))

    encoder.eval()
    classifier.eval()

    embeddings = extract_embeddings(test_texts, encoder, tokenizer, device=device)

    all_preds, all_labels, all_probs = [], [], []
    batch_size = 64

    for i in range(0, len(embeddings), batch_size):
        batch_embeds = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32).to(device)
        batch_labels = test_labels[i:i+batch_size]
        with torch.no_grad():
            outputs = classifier(batch_embeds)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
        all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"])
    conf_matrix = confusion_matrix(all_labels, all_preds)

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("\nClassification Report:\n" + report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n" + report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Neutral", "Positive"],
                yticklabels=["Negative", "Neutral", "Positive"])
    plt.title("Confusion Matrix (Test Data)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    logging.info(f"Confusion matrix saved to {confusion_matrix_path}")
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    logging.info("Inference and evaluation complete.")

if __name__ == "__main__":
    main()
