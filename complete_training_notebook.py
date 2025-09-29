#!/usr/bin/env python3
"""
Complete Transformer Training Notebook for Google Colab

This notebook contains all necessary code for training a Transformer model
with proper metrics calculation and visualization.
"""

import json
import os
import sys
import time
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install matplotlib seaborn tqdm nltk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data
nltk.download('punkt', quiet=True)

print("✅ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Configuration
CONFIG = {
    "data": {
        "max_length": 20,
        "batch_size": 32,
        "train_split": 0.8
    },
    "model": {
        "d_model": 512,
        "n_heads": 8,
        "n_encoder_layers": 6,
        "n_decoder_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_len": 5000
    },
    "training": {
        "epochs": 10,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0
    },
    "evaluation": {
        "enabled": True,
        "max_samples": 100
    }
}

print("📋 Configuration loaded:")
print(json.dumps(CONFIG, indent=2))

# Sample data for demonstration
SAMPLE_DATA = [
    {"src": "Hello world", "tgt": "Hallo Welt"},
    {"src": "Good morning", "tgt": "Guten Morgen"},
    {"src": "How are you?", "tgt": "Wie geht es dir?"},
    {"src": "Thank you", "tgt": "Danke"},
    {"src": "Goodbye", "tgt": "Auf Wiedersehen"},
    {"src": "What time is it?", "tgt": "Wie spät ist es?"},
    {"src": "Where is the station?", "tgt": "Wo ist der Bahnhof?"},
    {"src": "I love you", "tgt": "Ich liebe dich"},
    {"src": "Happy birthday", "tgt": "Alles Gute zum Geburtstag"},
    {"src": "See you later", "tgt": "Bis später"}
]

# Expand sample data
expanded_data = []
for _ in range(50):  # Create 500 samples
    for item in SAMPLE_DATA:
        expanded_data.append(item)

print(f"📊 Created {len(expanded_data)} training samples")

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # Embeddings
        src_emb = self.src_embedding(src) * np.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        
        # Add positional encoding
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Apply dropout
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # Create masks
        src_mask = self.transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=(src == 0),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(tgt == 0)
        )
        
        # Output projection
        output = self.output_projection(output)
        
        return output

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_length=20):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize and convert to indices
        src_tokens = item['src'].lower().split()
        tgt_tokens = item['tgt'].lower().split()
        
        # Convert to indices
        src_indices = [self.src_vocab.get(token, self.src_vocab['<UNK>']) for token in src_tokens]
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab['<UNK>']) for token in tgt_tokens]
        
        # Pad sequences
        src_padded = src_indices + [0] * (self.max_length - len(src_indices))
        tgt_padded = tgt_indices + [0] * (self.max_length - len(tgt_indices))
        
        # Truncate if too long
        src_padded = src_padded[:self.max_length]
        tgt_padded = tgt_padded[:self.max_length]
        
        return {
            'src': torch.tensor(src_padded, dtype=torch.long),
            'tgt': torch.tensor(tgt_padded, dtype=torch.long)
        }

# Build vocabularies
def build_vocab(data):
    src_tokens = set()
    tgt_tokens = set()
    
    for item in data:
        src_tokens.update(item['src'].lower().split())
        tgt_tokens.update(item['tgt'].lower().split())
    
    # Add special tokens
    src_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    tgt_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    # Add regular tokens
    for i, token in enumerate(sorted(src_tokens)):
        src_vocab[token] = i + 4
    
    for i, token in enumerate(sorted(tgt_tokens)):
        tgt_vocab[token] = i + 4
    
    return src_vocab, tgt_vocab

# Build vocabularies
src_vocab, tgt_vocab = build_vocab(expanded_data)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

print(f"📚 Vocabulary sizes:")
print(f"  Source: {src_vocab_size}")
print(f"  Target: {tgt_vocab_size}")

# Create datasets
train_size = int(len(expanded_data) * CONFIG['data']['train_split'])
train_data = expanded_data[:train_size]
val_data = expanded_data[train_size:]

train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, CONFIG['data']['max_length'])
val_dataset = TranslationDataset(val_data, src_vocab, tgt_vocab, CONFIG['data']['max_length'])

train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['data']['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['data']['batch_size'], shuffle=False)

print(f"📊 Dataset sizes:")
print(f"  Training: {len(train_dataset)}")
print(f"  Validation: {len(val_dataset)}")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    **CONFIG['model']
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['learning_rate'], 
                       weight_decay=CONFIG['training']['weight_decay'])

print(f"🚀 Model initialized on {device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training metrics storage
training_history = []

# Evaluation functions
def calculate_accuracy(predictions, targets, ignore_index=0):
    """Calculate accuracy ignoring padding tokens."""
    mask = targets != ignore_index
    correct = (predictions.argmax(dim=-1) == targets) & mask
    return correct.sum().float() / mask.sum().float()

def calculate_bleu_score(predictions, targets, tgt_vocab, ignore_index=0):
    """Calculate BLEU score for a batch."""
    if not CONFIG['evaluation']['enabled']:
        return 0.0
    
    bleu_scores = []
    smoothing = SmoothingFunction().method1
    
    # Create reverse vocabulary
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
    
    for i in range(predictions.size(0)):
        # Get predicted tokens
        pred_tokens = predictions[i].argmax(dim=-1).cpu().numpy()
        pred_tokens = [tgt_idx2word.get(idx, '<UNK>') for idx in pred_tokens if idx != ignore_index]
        
        # Get target tokens
        target_tokens = targets[i].cpu().numpy()
        target_tokens = [tgt_idx2word.get(idx, '<UNK>') for idx in target_tokens if idx != ignore_index]
        
        if len(pred_tokens) > 0 and len(target_tokens) > 0:
            try:
                bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

# Training function
def train_epoch():
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {len(training_history) + 1}")
    
    for batch in progress_bar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # Create input and target sequences
        tgt_input = tgt[:, :-1]  # Remove last token
        tgt_output = tgt[:, 1:]  # Remove first token
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(src, tgt_input)
        
        # Calculate loss
        loss = criterion(predictions.reshape(-1, predictions.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip_norm'])
        
        # Update parameters
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy = calculate_accuracy(predictions, tgt_output)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy.item():.4f}'
        })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_acc': total_accuracy / num_batches,
        'learning_rate': optimizer.param_groups[0]['lr']
    }

# Validation function
def validate():
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_bleu = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, desc="Validation")
        
        for batch in progress_bar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Create input and target sequences
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            predictions = model(src, tgt_input)
            
            # Calculate loss
            loss = criterion(predictions.reshape(-1, predictions.size(-1)), tgt_output.reshape(-1))
            
            # Calculate metrics
            accuracy = calculate_accuracy(predictions, tgt_output)
            bleu_score = calculate_bleu_score(predictions, tgt_output, tgt_vocab)
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_bleu += bleu_score
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'val_acc': f'{accuracy.item():.4f}',
                'bleu': f'{bleu_score:.4f}'
            })
    
    return {
        'val_loss': total_loss / num_batches,
        'val_acc': total_accuracy / num_batches,
        'bleu_score': total_bleu / num_batches
    }

# Visualization function
def plot_training_metrics():
    """Plot training metrics."""
    if not training_history:
        print("No training history available")
        return
    
    epochs = list(range(1, len(training_history) + 1))
    
    # Extract metrics
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    train_accs = [h['train_acc'] for h in training_history]
    val_accs = [h['val_acc'] for h in training_history]
    bleu_scores = [h.get('bleu_score', 0) for h in training_history]
    learning_rates = [h['learning_rate'] for h in training_history]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_accs, 'g-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'orange', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: BLEU scores
    ax3 = axes[1, 0]
    ax3.plot(epochs, bleu_scores, 'purple', label='BLEU Score', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('BLEU Score')
    ax3.set_title('BLEU Score Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    ax4 = axes[1, 1]
    ax4.plot(epochs, learning_rates, 'brown', label='Learning Rate', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main training loop
print("🚀 Starting training...")
print("=" * 80)

for epoch in range(CONFIG['training']['epochs']):
    print(f"\n📊 Epoch {epoch + 1}/{CONFIG['training']['epochs']}")
    print("-" * 50)
    
    # Train
    train_metrics = train_epoch()
    
    # Validate
    val_metrics = validate()
    
    # Combine metrics
    epoch_metrics = {
        'epoch': epoch + 1,
        **train_metrics,
        **val_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    training_history.append(epoch_metrics)
    
    # Print epoch summary
    print(f"✅ Epoch {epoch + 1} completed:")
    print(f"   Train Loss: {train_metrics['train_loss']:.4f}")
    print(f"   Train Acc:  {train_metrics['train_acc']:.4f}")
    print(f"   Val Loss:   {val_metrics['val_loss']:.4f}")
    print(f"   Val Acc:    {val_metrics['val_acc']:.4f}")
    print(f"   BLEU Score: {val_metrics['bleu_score']:.4f}")
    print(f"   Learning Rate: {train_metrics['learning_rate']:.6f}")

print("\n🎉 Training completed!")
print("=" * 80)

# Plot final metrics
plot_training_metrics()

# Save results
results = {
    'config': CONFIG,
    'vocab_sizes': {'src': src_vocab_size, 'tgt': tgt_vocab_size},
    'training_history': training_history,
    'final_metrics': training_history[-1] if training_history else {}
}

# Save to file
with open('training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("📁 Results saved to training_results.json")

# Create download link
from google.colab import files
files.download('training_results.json')

print("✅ Training results downloaded!")
print("\n📊 Final Summary:")
if training_history:
    final = training_history[-1]
    print(f"   Final Train Loss: {final['train_loss']:.4f}")
    print(f"   Final Train Acc:  {final['train_acc']:.4f}")
    print(f"   Final Val Loss:   {final['val_loss']:.4f}")
    print(f"   Final Val Acc:    {final['val_acc']:.4f}")
    print(f"   Final BLEU Score: {final['bleu_score']:.4f}")

print("\n🎯 All metrics are now properly calculated and visualized!")
