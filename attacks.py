

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def unwrap_model(model):
    # Opacus wraps in GradSampleModule, stores original in _module
    if hasattr(model, '_module'):
        return model._module
    return model


def run_mia_attack(model, train_dataset, test_dataset, config, device):
    model = unwrap_model(model)
    model.eval()

    #  loss function
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Unpack data
    train_seqs, train_labs, _, train_lens = train_dataset
    test_seqs, test_labs, _, test_lens = test_dataset

    # Sample samw size as train  and test 
    n_samples = min(len(train_seqs), len(test_seqs), config.get('mia_samples', 2000))

    seed = config.get('seed', 42)
    np.random.seed(seed)

    # Sample members
    member_idx = np.random.choice(len(train_seqs), n_samples, replace=False)

    # Sample non-members
    nonmember_idx = np.random.choice(len(test_seqs), n_samples, replace=False)

    # Prepare member batch
    mem_seqs = torch.LongTensor([train_seqs[i] for i in member_idx])
    mem_labs = torch.FloatTensor([train_labs[i] for i in member_idx])
    mem_lens = torch.LongTensor([train_lens[i] for i in member_idx])

    # Sort members by length
    mem_lens, perm = mem_lens.sort(0, descending=True)
    mem_seqs = mem_seqs[perm].to(device)
    mem_labs = mem_labs[perm].to(device)
    mem_lens = mem_lens.to(device)

    # Prepare non-member batch
    nonmem_seqs = torch.LongTensor([test_seqs[i] for i in nonmember_idx])
    nonmem_labs = torch.FloatTensor([test_labs[i] for i in nonmember_idx])
    nonmem_lens = torch.LongTensor([test_lens[i] for i in nonmember_idx])

    # Sort non-members
    nonmem_lens, perm2 = nonmem_lens.sort(0, descending=True)
    nonmem_seqs = nonmem_seqs[perm2].to(device)
    nonmem_labs = nonmem_labs[perm2].to(device)
    nonmem_lens = nonmem_lens.to(device)

    # Compute losses
    with torch.no_grad():
        mem_out = model(mem_seqs, mem_lens)
        mem_loss = criterion(mem_out, mem_labs).cpu().numpy()

        nonmem_out = model(nonmem_seqs, nonmem_lens)
        nonmem_loss = criterion(nonmem_out, nonmem_labs).cpu().numpy()

    # Attack: use negative loss as score (lower loss -> higher score -> member)
    member_scores = -mem_loss
    nonmember_scores = -nonmem_loss
    all_scores = np.concatenate([member_scores, nonmember_scores])
    all_labels = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])

    # Check for NaN
    if np.isnan(all_scores).any():
        print("  MIA: WARNING - NaN in scores")
        return {'mia_auc': 0.5, 'mia_samples': n_samples}

    # ROC-AUC
    auc = roc_auc_score(all_labels, all_scores)
    print(f"  MIA: auc={auc:.4f}")

    return {'mia_auc': auc, 'mia_samples': n_samples}


def run_attribute_inference_attack(model, dataset, config, device):
    
    model = unwrap_model(model)
    model.eval()

    seqs, labels, orig_lens, clip_lens = dataset

    # Synthetic attribute: long vs short review  -> binary
    median_len = np.median(orig_lens)
    attrs = (np.array(orig_lens) > median_len).astype(int)

    # Prepare data for model
    seq_tensor = torch.LongTensor(seqs)
    len_tensor = torch.LongTensor(clip_lens)

    # Sort for packing
    len_tensor, sort_idx = len_tensor.sort(0, descending=True)
    seq_tensor = seq_tensor[sort_idx].to(device)
    len_tensor = len_tensor.to(device)
    # Permute attributes too
    attrs = attrs[sort_idx.numpy()]

    # Extract representations
    with torch.no_grad():
        reps = model.get_representation(seq_tensor, len_tensor)
        reps = reps.cpu().numpy()

    # Train attack classifier
    test_frac = config.get('attr_test_size', 0.3)
    seed = config.get('seed', 42)

    X_tr, X_te, y_tr, y_te = train_test_split(
        reps, attrs,
        test_size=test_frac,
        random_state=seed,
        stratify=attrs
    )

    # Logistic regression
    classifier = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    classifier.fit(X_tr, y_tr)

    # Test accuracy
    preds = classifier.predict(X_te)
    acc = accuracy_score(y_te, preds)

    baseline = 0.5
    advantage = acc - baseline

    print(f"  Attr inf: acc={acc:.4f}, advantage={advantage:.4f}")

    return {
        'attr_inf_accuracy': acc,
        'attr_inf_baseline': baseline,
        'attr_inf_advantage': advantage
    }
