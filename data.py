"""
Data loading and preprocessing for IMDb.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import numpy as np
from typing import Dict, List, Tuple
import random


class IMDbDataset(Dataset):
    """Dataset for IMDb sequences."""

    def __init__(self, sequences, labels, original_lengths=None, clipped_lengths=None):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.original_lengths = original_lengths
        # Clipped lengths needed for LSTM packing
        if clipped_lengths is not None:
            self.clipped_lengths = torch.LongTensor(clipped_lengths)
        else:
            self.clipped_lengths = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.clipped_lengths is not None:
            return self.sequences[idx], self.labels[idx], self.clipped_lengths[idx]
        else:
            return self.sequences[idx], self.labels[idx]


def load_and_split_imdb(config):
    """Load IMDb and create train/val/test splits."""
    # Load dataset
    dataset = load_dataset("imdb")

    # Set seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)

    # Split sizes
    n_train = config['train_size']
    n_val = config['val_size']
    n_test = config['test_size']

    # Use HF train split for our train+val
    total_train_available = len(dataset['train'])
    train_indices = list(range(total_train_available))
    random.shuffle(train_indices)

    # Take first n_train for training
    idx_train = train_indices[:n_train]
    # Take next n_val for validation
    idx_val = train_indices[n_train:n_train + n_val]

    # Use HF test split for our test
    total_test_available = len(dataset['test'])
    test_indices = list(range(total_test_available))
    random.shuffle(test_indices)
    idx_test = test_indices[:n_test]

    # Create splits
    train_data = dataset['train'].select(idx_train)
    val_data = dataset['train'].select(idx_val)
    test_data = dataset['test'].select(idx_test)

    return train_data, val_data, test_data


def tokenize_text(text: str) -> List[str]:
    """Tokenize text: lowercase and split."""
    return text.lower().split()


def build_vocab_and_sequences(datasets, config):
    """Build vocab from train data and convert all datasets to sequences."""
    train_data, val_data, test_data = datasets

    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'

    # Build vocabulary from training set only
    word_counts = Counter()
    for sample in train_data:
        tokens = tokenize_text(sample['text'])
        word_counts.update(tokens)

    # Keep top words (reserve 2 slots for special tokens)
    max_vocab = config['max_vocab_size']
    top_words = word_counts.most_common(max_vocab - 2)

    # Create word->index mapping
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, (word, count) in enumerate(top_words, start=2):
        word2idx[word] = i

    # Create reverse mapping
    idx2word = {i: w for w, i in word2idx.items()}

    vocab_size = len(word2idx)

    vocab = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': vocab_size,
        'pad_idx': 0,
        'unk_idx': 1
    }

    # Convert text to sequences
    max_len = config['max_seq_len']
    pad_idx = word2idx[PAD_TOKEN]
    unk_idx = word2idx[UNK_TOKEN]

    def convert_to_sequences(data):
        """Convert dataset to padded sequences."""
        seqs = []
        labs = []
        orig_lens = []
        clip_lens = []

        for item in data:
            text = item['text']
            label = item['label']

            # Tokenize
            tokens = tokenize_text(text)
            orig_lens.append(len(tokens))

            # Convert words to indices
            indices = []
            for tok in tokens:
                if tok in word2idx:
                    indices.append(word2idx[tok])
                else:
                    indices.append(unk_idx)

            # Clip or pad to max_len
            actual_len = min(len(indices), max_len)
            clip_lens.append(actual_len)

            if len(indices) > max_len:
                # Truncate
                indices = indices[:max_len]
            else:
                # Pad
                padding_needed = max_len - len(indices)
                indices = indices + [pad_idx] * padding_needed

            seqs.append(indices)
            labs.append(label)

        return seqs, labs, orig_lens, clip_lens

    # Process all splits
    train_seqs, train_labs, train_orig, train_clip = convert_to_sequences(train_data)
    val_seqs, val_labs, val_orig, val_clip = convert_to_sequences(val_data)
    test_seqs, test_labs, test_orig, test_clip = convert_to_sequences(test_data)

    datasets_as_tensors = {
        'train': (train_seqs, train_labs, train_orig, train_clip),
        'val': (val_seqs, val_labs, val_orig, val_clip),
        'test': (test_seqs, test_labs, test_orig, test_clip)
    }

    return vocab, datasets_as_tensors


def collate_fn(batch):
    """Collate batch and sort by length for LSTM packing."""
    # Unpack batch
    sequences, labels, lengths = zip(*batch)

    # Convert to tensors
    seq_tensor = torch.stack(sequences)
    lab_tensor = torch.stack(labels)
    len_tensor = torch.stack(lengths)

    # Sort by length descending (required for pack_padded_sequence)
    sorted_lengths, sort_indices = len_tensor.sort(0, descending=True)
    sorted_seqs = seq_tensor[sort_indices]
    sorted_labs = lab_tensor[sort_indices]

    return sorted_seqs, sorted_labs, sorted_lengths


def create_dataloaders(datasets, config, dp: bool = False):
    """Create DataLoaders for train/val/test."""
    # Create datasets
    train_dataset = IMDbDataset(*datasets['train'])
    val_dataset = IMDbDataset(*datasets['val'])
    test_dataset = IMDbDataset(*datasets['test'])

    # Batch size depends on DP or not
    if dp:
        batch_size = config['batch_size_dp']
    else:
        batch_size = config['batch_size_non_dp']

    # Train loader - use shuffling and more workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    # Val loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
