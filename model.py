

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from opacus.layers import DPLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentLSTM(nn.Module):

    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=1, dropout=0.3, padding_idx=0, freeze_embeddings=True):
        super(SentimentLSTM, self).__init__()

        # Extract dimensions
        vocab_size, emb_dim = embedding_matrix.shape

        #   embedding layer with pretrained weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )


        # output is 2*hidden_dim
        dropout_rate = dropout if num_layers > 1 else 0.0
        self.lstm = DPLSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):

        # Embed input tokens
        emb = self.embedding(x)

        # Pack sequences for efficient LSTM processing (skip padding)
        packed_input = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # Run LSTM
        packed_out, (hidden_states, cell_states) = self.lstm(packed_input)

        # Get final hidden states from both directions
        # For bidirectional: h_n[-2] is forward, h_n[-1] is backward
        h_forward = hidden_states[-2]
        h_backward = hidden_states[-1]
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        # Classify
        out = self.fc(h_combined)

        return out.squeeze(1)

    def get_representation(self, x, lengths):
        
        with torch.no_grad():
            # Run through embedding and LSTM
            x_emb = self.embedding(x)
            packed_seq = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
            _, (h, c) = self.lstm(packed_seq)

            # Concatenate bidirectional hidden states
            representation = torch.cat((h[-2], h[-1]), dim=1)

        return representation


def load_glove_for_vocab(glove_path, word2idx, embedding_dim=100):

    vocab_size = len(word2idx)

    # embedding matrix
    emb_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype(np.float32)

    # Load GloVe vectors
    found_count = 0
    skipped = 0

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                if word not in word2idx:
                    continue

                # Parse vector
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    skipped += 1
                    continue

                # Check dimension matches
                if len(vec) != embedding_dim:
                    skipped += 1
                    continue

                # Store in matrix
                idx = word2idx[word]
                emb_matrix[idx] = vec
                found_count += 1

            except (ValueError, IndexError):
                skipped += 1
                continue

    # Padding should be zero vector
    if '<PAD>' in word2idx:
        pad_idx = word2idx['<PAD>']
        emb_matrix[pad_idx] = np.zeros(embedding_dim, dtype=np.float32)

    return emb_matrix


def build_model(vocab_size, embedding_matrix, config):
    # Sanity check dimensions
    assert embedding_matrix.shape[0] == vocab_size, "Embedding matrix vocab size mismatch"

    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 1)
    dropout = config.get('dropout', 0.3)
    pad_idx = config.get('pad_idx', 0)
    freeze_emb = config.get('freeze_embeddings', True)

    model = SentimentLSTM(
        embedding_matrix=embedding_matrix,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=pad_idx,
        freeze_embeddings=freeze_emb
    )

    return model
