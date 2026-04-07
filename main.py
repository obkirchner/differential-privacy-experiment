import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from pathlib import Path
import random
import argparse
from copy import deepcopy

from data import load_and_split_imdb, build_vocab_and_sequences, create_dataloaders
from model import build_model, load_glove_for_vocab
from attacks import run_mia_attack, run_attribute_inference_attack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic - important for DP
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y, lengths in loader:
        # Move to device
        batch_x, batch_y, lengths = batch_x.to(device), batch_y.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x, lengths)
        loss = criterion(outputs, batch_y)

        # training divergence?
        if torch.isnan(loss):
            raise ValueError("NaN loss - training diverged (check learning rate)")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    #  average
    avg_loss = total_loss / num_batches
    if np.isnan(avg_loss):
        raise ValueError("NaN in average loss")

    return avg_loss


def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch_x, batch_y, lengths in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            lengths = lengths.to(device)

            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Predictions: apply sigmoid and threshold
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Collect predictions and labels
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(batch_y.cpu().numpy())

    # Convert to numpy arrays for sklearn
    preds_array = np.array(preds_list)
    labels_array = np.array(labels_list)

    # Calculate metrics
    acc = accuracy_score(labels_array, preds_array)
    prec = precision_score(labels_array, preds_array, zero_division=0)
    rec = recall_score(labels_array, preds_array, zero_division=0)
    f1 = f1_score(labels_array, preds_array, zero_division=0)

    return {
        'loss': total_loss / len(loader),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

   #Returns model, epsilon, delta
def train_model(model, train_loader, val_loader, config, dp_config, device, use_dp=False, noise_multiplier=0.0):
 

    criterion = nn.BCEWithLogitsLoss()

    # Setup optimizer -->SGD for both baseline and DP
    lr = config['learning_rate']
    momentum = config['momentum']
    weight_decay = config.get('weight_decay', 0.0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    epsilon, delta = None, None
    privacy_engine = None

    # Apply DP 
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=dp_config['max_grad_norm']
        )
        delta = dp_config['delta']

    # Training loop 
    num_epochs = config['num_epochs']
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(num_epochs):
        # Train and evaluate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Handle Opacus wrapping
            if hasattr(model, '_module'):
                best_model_state = deepcopy(model._module.state_dict())
            else:
                best_model_state = deepcopy(model.state_dict())

        # Print progress
        if use_dp:
            eps_current = privacy_engine.get_epsilon(delta)
            print(f"  epoch {epoch+1}: acc={val_acc:.4f}, f1={val_f1:.4f}, eps={eps_current:.2f}")
        else:
            print(f"  epoch {epoch+1}: acc={val_acc:.4f}, f1={val_f1:.4f}")

    # Load best checkpoint
    if best_model_state is not None:
        if hasattr(model, '_module'):
            model._module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)

    # Final epsilon
    if use_dp:
        epsilon = privacy_engine.get_epsilon(delta)

    return model, epsilon, delta


def main():
    parser = argparse.ArgumentParser(description='DP-LSTM Privacy Experiment')
    parser.add_argument('--mode', type=str, choices=['debug', 'full'], default=None,
                        help='Run mode (debug or full)')
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Determine mode (from args or config default)
    mode = args.mode if args.mode else config['modes']['default']
    mode_cfg = config['modes'][mode]

    # Set seed
    seed = config['seed']
    set_seed(seed)

    # Create output dirs
    Path('models').mkdir(exist_ok=True)

    # Build data config (flatten nested config)
    data_config = {
        'seed': seed,
        'train_size': mode_cfg['train_size'],
        'val_size': mode_cfg['val_size'],
        'test_size': mode_cfg['test_size'],
        'max_vocab_size': config['data']['max_vocab_size'],
        'max_seq_len': config['data']['max_seq_len'],
        'batch_size_non_dp': config['training']['batch_size_non_dp'],
        'batch_size_dp': config['training']['batch_size_dp']
    }

    # Training config
    training_config = config['training'].copy()
    training_config['num_epochs'] = mode_cfg['num_epochs']  # Override from mode

    # Model config
    model_config = {
        'hidden_dim': config['model']['hidden_dim'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'pad_idx': config['model']['padding_idx'],
        'freeze_embeddings': config['model']['freeze_embeddings']
    }

    # Load data
    datasets = load_and_split_imdb(data_config)
    vocab, datasets_tensors = build_vocab_and_sequences(datasets, data_config)
    vocab_size = vocab['vocab_size']

    # Load GloVe
    glove_path = config['model']['glove']['path']
    embedding_matrix = load_glove_for_vocab(glove_path, vocab['word2idx'],
                                             config['model']['embedding_dim'])

    print("IMDb dataset loaded")

    # Train all experiments
    for exp_config in config['experiments']:
        exp_name = exp_config['name']
        use_dp = exp_config['dp']
        noise = exp_config['noise_multiplier']

        print(f"\n{exp_name}:")

        # Build model
        model = build_model(vocab_size, embedding_matrix, model_config)
        model = model.to(device)

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            datasets_tensors, data_config, dp=use_dp)

        # Experiment-specific training config
        exp_train_cfg = training_config.copy()
        if 'learning_rate' in exp_config:
            exp_train_cfg['learning_rate'] = exp_config['learning_rate']
        if 'momentum' in exp_config:
            exp_train_cfg['momentum'] = exp_config['momentum']

        # Train
        trained_model, epsilon, delta = train_model(
            model, train_loader, val_loader,
            exp_train_cfg, config['dp'],
            use_dp=use_dp, noise_multiplier=noise, device=device
        )

        # Test evaluation
        criterion = nn.BCEWithLogitsLoss()
        test_metrics = evaluate(trained_model, test_loader, criterion, device)
        print(f"Test: acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}")

        # Privacy attacks
        mia_results = run_mia_attack(
            trained_model,
            datasets_tensors['train'],
            datasets_tensors['test'],
            config, device
        )
        attr_results = run_attribute_inference_attack(
            trained_model,
            datasets_tensors['test'],
            config, device
        )

        # Save checkpoint
        model_path = f"models/{exp_name}.pt"
        torch.save(trained_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
