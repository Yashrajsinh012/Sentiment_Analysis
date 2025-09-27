#scripts/model.py

import torch
import torch.nn as nn
from transformers import BertModel

class BERT_BiLSTM_Attention(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_size=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bilstm = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers=2,
                              batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.embedding_fc = nn.Linear(hidden_size * 2, hidden_size * 2)

    def attention_layer(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        return torch.sum(attn_weights * lstm_output, dim=1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.bilstm(bert_output.last_hidden_state)
        attn_output = self.attention_layer(lstm_out)
        return self.embedding_fc(attn_output)


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.fc(x)

