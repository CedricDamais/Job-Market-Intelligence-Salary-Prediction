from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import math


import random
from torch.utils.data import IterableDataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
import math
import torch
import torch.nn as nn
import numpy as np
import logging
import json
import os
import sys


class HybridFFNN(nn.Module):
    def __init__(self, vocab, hidden_size, dropout, num_layers, context_size, lr, pretrained_embeddings=None):
        super().__init__()
        self.vocab = vocab
        self.rev_vocab = {i: t for t, i in vocab.items()}
        self.lr = lr
        self.N = context_size
        self.context_size = context_size
        self.vocab_size = len(vocab)
        self.embedding_dim = (
            pretrained_embeddings.shape[1] if pretrained_embeddings is not None else 384
        )
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if pretrained_embeddings is not None:
            w = torch.FloatTensor(pretrained_embeddings) if isinstance(pretrained_embeddings, np.ndarray) else pretrained_embeddings
            self.embedding.weight.data.copy_(w)
        layers = []
        input_dim = self.embedding_dim * self.context_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_size, self.vocab_size))
        self.model = nn.Sequential(*layers)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        flat = embedded.view(embedded.size(0), -1)
        return self.model(flat)

    def metrics(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = self.embedding.weight.data
        return {'total_params': total, 'trainable_params': trainable,
                'emb_mean': emb.mean().item(), 'emb_std': emb.std().item()}

    def fit(self, train_df, val_df=None, batch_size=10000, epochs=10, device='cuda',
            optimizer_type='adamw', momentum=0.9, weight_decay=1e-2,
            scheduler_type='onecycle'):
        """Train with LR schedules and momentum-based optimizers."""
        self.to(device)
        N = self.context_size
        # compute steps
        total_windows = sum(max(len(row['merged_tokenized']) - N, 0) for _, row in train_df.iterrows())
        steps_per_epoch = math.ceil(total_windows / batch_size)
        # dataloader
        dataset = ContextWindowDataset(train_df, self.vocab, N)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda b: (
            torch.stack([x[0] for x in b]), torch.LongTensor([x[1] for x in b])
        ))
        # optimizer
        if optimizer_type=='sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type=='adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler
        scheduler=None; step_every=None
        if scheduler_type=='onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.3, anneal_strategy='cos')
            step_every='batch'
        elif scheduler_type=='cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            step_every='epoch'
        elif scheduler_type=='plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
            step_every='val'
        loss_fn=nn.CrossEntropyLoss()
        # storage for logging
        loss_values = []
        # train loop Frostlord123
        for epoch in range(1,epochs+1):
            self.train(); total_loss=0.0; count=0
            logging.info(f"Epoch {epoch}/{epochs} start ({optimizer_type}+{scheduler_type})")
            running_loss_100 = 0.0
            for batch_idx, (inputs, targets) in enumerate(loader, start=1):
                inputs,targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = self(inputs)
                loss = loss_fn(logits, targets)
                loss.backward(); optimizer.step()
                if step_every=='batch': scheduler.step()
                # update losses
                total_loss += loss.item(); count += 1
                running_loss_100 += loss.item()
                # log every 100 batches
                if batch_idx % 100 == 0:
                    avg100 = running_loss_100 / 100
                    loss_values.append(avg100)
                    logging.info(f"Epoch {epoch}, Batch {batch_idx}: Avg100 Loss = {avg100:.4f}")
                    running_loss_100 = 0.0
            avg=total_loss/count if count else float('nan')
            msg=f"Epoch {epoch} train loss: {avg:.4f}"
            if val_df is not None:
                vppl=BenchmarkModel(val_df,self).perplexity()
                msg+=f", val ppl: {vppl:.2f}"
                if scheduler_type=='plateau': scheduler.step(vppl)
            logging.info(msg)
            if step_every=='epoch' and scheduler is not None: scheduler.step()
        return loss_values

                

class BenchmarkModel:
    def __init__(self, test_df, model_to_test):
        """Initialize with test DataFrame and a trained model."""
        self.test_df = test_df
        self.model = model_to_test.to(next(model_to_test.parameters()).device)
        self.context_size = self.model.context_size
        self.vocab = self.model.vocab
        self.rev_vocab = {i: t for t, i in self.vocab.items()}

    def predict_job_description(self, context_window, max_length=50, temperature=1.0):
        """Generate a sequence autoregressively from a starting context."""
        self.model.eval()
        device = next(self.model.parameters()).device
        tokens = context_window.copy()
        
        for _ in range(max_length):
            idxs = torch.LongTensor([
                self.vocab.get(t, self.vocab.get("__UNK__")) for t in tokens[-self.context_size:]
            ]).unsqueeze(0).to(device)
            
            logits = self.model(idxs).squeeze(0) / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Set probability of UNK token to 0
            unk_idx = self.vocab.get("__UNK__", -1)
            if unk_idx >= 0:
                probs[unk_idx] = 0
                # Renormalize if needed
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    # If all probs are zero, use uniform distribution
                    probs = torch.ones_like(probs) / len(probs)
            
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_tok = self.rev_vocab.get(next_idx, "<unk>")
            tokens.append(next_tok)
            
            # Optional: stop if end token is generated
            if next_tok == "__END__" or next_tok == "<END>":
                break
                
        return tokens

    def perplexity(self):
        """Compute perplexity on the test set."""
        loss_fn = nn.CrossEntropyLoss()
        dataset = ContextWindowDataset(self.test_df, self.vocab, self.context_size)
        
        # Modified collate function to handle the dataset output correctly
        def collate_fn(batch):
            contexts = torch.stack([x[0] for x in batch])  # Get context_idxs
            targets = torch.LongTensor([x[1] for x in batch])  # Get target_idx
            return contexts, targets
        
        loader = DataLoader(
            dataset, 
            batch_size=32, 
            collate_fn=collate_fn
        )
        
        total_loss, count = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(next(self.model.parameters()).device)
                targets = targets.to(next(self.model.parameters()).device)
                logits = self.model(inputs)
                loss = loss_fn(logits, targets)
                total_loss += loss.item()
                count += 1
                
        avg_loss = total_loss / count if count else float('nan')
        return math.exp(avg_loss)

    def bleu_score(self):
        """Compute corpus BLEU over next-token predictions."""
        dataset = ContextWindowDataset(self.test_df, self.vocab, self.context_size)
        refs, hyps = [], []
        for ctx_idxs, tgt_idx in dataset:  # Only unpack the two values that are actually returned
            pred_idx = self.model(ctx_idxs.unsqueeze(0).to(next(self.model.parameters()).device)).argmax(dim=-1).item()
            pred_token = self.rev_vocab.get(pred_idx, "<unk>")
            target_token = self.rev_vocab.get(tgt_idx, "<unk>")
            hyps.append([pred_token])
            refs.append([[target_token]])
        return corpus_bleu(refs, hyps)

    def rouge_score(self):
        """Compute average ROUGE-1 and ROUGE-L F1 over single-token predictions."""
        scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
        total1, totalL, count = 0.0, 0.0, 0
        dataset = ContextWindowDataset(self.test_df, self.vocab, self.context_size)
        for ctx_idxs, tgt_idx in dataset:  # Only unpack the two values that are actually returned
            pred_idx = self.model(ctx_idxs.unsqueeze(0).to(next(self.model.parameters()).device)).argmax(dim=-1).item()
            pred_token = self.rev_vocab.get(pred_idx, "<unk>")
            target_token = self.rev_vocab.get(tgt_idx, "<unk>")
            scores = scorer.score(target_token, pred_token)
            total1 += scores['rouge1'].fmeasure
            totalL += scores['rougeL'].fmeasure
            count += 1
        return {'rouge1': total1/count if count else 0.0,
                'rougeL': totalL/count if count else 0.0}
    

def load_vocab(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Get the project root path
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.append(PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
    if PROJECT_ROOT not in sys.path:
         sys.path.append(PROJECT_ROOT)

# Load vocabulary with correct path
VOCAB_PATH = os.path.join(PROJECT_ROOT, "models/src/generation/data/processed/vocab.json")
tokenized_vocab = load_vocab(VOCAB_PATH)

def load_embeddings(filepath):
    """Load pretrained embeddings from a .npy file."""
    return np.load(filepath)

# Load embeddings with correct path
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "models/src/generation/data/processed/pretrained_embeddings.npy")
pretrained_embeddings = load_embeddings(EMBEDDINGS_PATH)
