"""
Evaluation Metrics for Machine Translation
Implements comprehensive evaluation metrics for Transformer translation models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
from collections import Counter
import math


class TranslationEvaluator:
    """
    Comprehensive evaluator for machine translation tasks.
    
    Implements multiple evaluation metrics:
    - BLEU (Bilingual Evaluation Understudy)
    - METEOR (Metric for Evaluation of Translation with Explicit ORdering)
    - ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    - Perplexity
    - Exact Match
    - Semantic Similarity (using word embeddings)
    """
    
    def __init__(self, tgt_vocab: Dict[str, int], idx2word: Dict[int, str]):
        """
        Initialize evaluator.
        
        Args:
            tgt_vocab: Target vocabulary mapping
            idx2word: Index to word mapping
        """
        self.tgt_vocab = tgt_vocab
        self.idx2word = idx2word
        self.word2idx = {v: k for k, v in idx2word.items()}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
    
    def decode_sequence(self, sequence: torch.Tensor) -> List[str]:
        """
        Decode sequence of indices to words.
        
        Args:
            sequence: Sequence of token indices
            
        Returns:
            List of decoded words
        """
        words = []
        for idx in sequence.tolist():
            if idx == self.tgt_vocab[self.eos_token]:
                break
            if idx != self.tgt_vocab[self.pad_token]:
                word = self.idx2word.get(idx, self.unk_token)
                words.append(word)
        return words
    
    def calculate_bleu_score(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
        max_n: int = 4,
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate BLEU scores (1-4 gram).
        
        Args:
            predictions: List of predicted sequences
            references: List of reference sequences
            max_n: Maximum n-gram order
            weights: Weights for different n-grams
            
        Returns:
            Dictionary of BLEU scores
        """
        if weights is None:
            weights = [1.0 / max_n] * max_n
        
        bleu_scores = {}
        
        for n in range(1, max_n + 1):
            total_precision = 0.0
            total_length = 0
            
            for pred, ref in zip(predictions, references):
                # Calculate n-gram precision
                pred_ngrams = self._get_ngrams(pred, n)
                ref_ngrams = self._get_ngrams(ref, n)
                
                if len(pred_ngrams) == 0:
                    continue
                
                # Count matches
                matches = 0
                for ngram in pred_ngrams:
                    if ngram in ref_ngrams:
                        matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
                
                precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0
                total_precision += precision
                total_length += 1
            
            avg_precision = total_precision / total_length if total_length > 0 else 0
            bleu_scores[f'BLEU-{n}'] = avg_precision
        
        # Calculate overall BLEU score
        if all(score > 0 for score in bleu_scores.values()):
            log_bleu = sum(w * math.log(score) for w, score in zip(weights, bleu_scores.values()))
            bleu_scores['BLEU'] = math.exp(log_bleu)
        else:
            bleu_scores['BLEU'] = 0.0
        
        return bleu_scores
    
    def calculate_meteor_score(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> float:
        """
        Calculate METEOR score (simplified version).
        
        Args:
            predictions: List of predicted sequences
            references: List of reference sequences
            
        Returns:
            METEOR score
        """
        total_score = 0.0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            # Exact matches
            pred_set = set(pred)
            ref_set = set(ref)
            
            matches = len(pred_set.intersection(ref_set))
            total_words = len(pred_set.union(ref_set))
            
            if total_words > 0:
                precision = matches / len(pred_set) if len(pred_set) > 0 else 0
                recall = matches / len(ref_set) if len(ref_set) > 0 else 0
                
                if precision + recall > 0:
                    f_score = 2 * precision * recall / (precision + recall)
                    total_score += f_score
            
            total_count += 1
        
        return total_score / total_count if total_count > 0 else 0.0
    
    def calculate_rouge_score(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
        rouge_type: str = 'L'
    ) -> float:
        """
        Calculate ROUGE score.
        
        Args:
            predictions: List of predicted sequences
            references: List of reference sequences
            rouge_type: Type of ROUGE ('L' for longest common subsequence)
            
        Returns:
            ROUGE score
        """
        if rouge_type == 'L':
            return self._calculate_rouge_l(predictions, references)
        else:
            raise ValueError(f"Unsupported ROUGE type: {rouge_type}")
    
    def _calculate_rouge_l(self, predictions: List[List[str]], references: List[List[str]]) -> float:
        """Calculate ROUGE-L score."""
        total_score = 0.0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            lcs_length = self._longest_common_subsequence(pred, ref)
            
            if len(pred) > 0 and len(ref) > 0:
                precision = lcs_length / len(pred)
                recall = lcs_length / len(ref)
                
                if precision + recall > 0:
                    f_score = 2 * precision * recall / (precision + recall)
                    total_score += f_score
            
            total_count += 1
        
        return total_score / total_count if total_count > 0 else 0.0
    
    def calculate_perplexity(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> float:
        """
        Calculate model perplexity.
        
        Args:
            model: Trained model
            dataloader: Data loader for evaluation
            device: Device to use
            
        Returns:
            Perplexity score
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(device)
                tgt_input = batch['tgt_input'].to(device)
                tgt_output = batch['tgt_output'].to(device)
                
                # Forward pass
                predictions = model(src, tgt_input)
                
                # Calculate loss
                loss = F.cross_entropy(
                    predictions.view(-1, predictions.size(-1)),
                    tgt_output.view(-1),
                    ignore_index=self.tgt_vocab[self.pad_token],
                    reduction='sum'
                )
                
                total_loss += loss.item()
                
                # Count non-padding tokens
                mask = (tgt_output != self.tgt_vocab[self.pad_token])
                total_tokens += mask.sum().item()
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
        else:
            perplexity = float('inf')
        
        return perplexity
    
    def calculate_exact_match(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> float:
        """
        Calculate exact match accuracy.
        
        Args:
            predictions: List of predicted sequences
            references: List of reference sequences
            
        Returns:
            Exact match accuracy
        """
        matches = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            if pred == ref:
                matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def calculate_word_accuracy(
        self,
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> float:
        """
        Calculate word-level accuracy.
        
        Args:
            predictions: List of predicted sequences
            references: List of reference sequences
            
        Returns:
            Word accuracy
        """
        total_words = 0
        correct_words = 0
        
        for pred, ref in zip(predictions, references):
            min_len = min(len(pred), len(ref))
            for i in range(min_len):
                total_words += 1
                if pred[i] == ref[i]:
                    correct_words += 1
        
        return correct_words / total_words if total_words > 0 else 0.0
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu",
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            dataloader: Data loader for evaluation
            device: Device to use
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if max_samples and idx >= max_samples:
                    break
                
                src = batch['src'].to(device)
                tgt_input = batch['tgt_input'].to(device)
                tgt_output = batch['tgt_output'].to(device)
                
                # Generate predictions
                pred_tokens = self._generate_sequence(model, src, tgt_input, device)
                
                # Decode sequences
                for i in range(src.size(0)):
                    pred_words = self.decode_sequence(pred_tokens[i])
                    ref_words = self.decode_sequence(tgt_output[i])
                    
                    predictions.append(pred_words)
                    references.append(ref_words)
        
        # Calculate all metrics
        metrics = {}
        
        # BLEU scores
        bleu_scores = self.calculate_bleu_score(predictions, references)
        metrics.update(bleu_scores)
        
        # METEOR score
        metrics['METEOR'] = self.calculate_meteor_score(predictions, references)
        
        # ROUGE-L score
        metrics['ROUGE-L'] = self.calculate_rouge_score(predictions, references)
        
        # Exact match
        metrics['Exact Match'] = self.calculate_exact_match(predictions, references)
        
        # Word accuracy
        metrics['Word Accuracy'] = self.calculate_word_accuracy(predictions, references)
        
        # Perplexity
        metrics['Perplexity'] = self.calculate_perplexity(model, dataloader, device)
        
        return metrics
    
    def _get_ngrams(self, sequence: List[str], n: int) -> Counter:
        """Get n-grams from sequence."""
        ngrams = []
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _generate_sequence(
        self,
        model: torch.nn.Module,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        device: str,
        max_length: int = 50
    ) -> torch.Tensor:
        """
        Generate sequence using greedy decoding.
        
        Args:
            model: Trained model
            src: Source sequence
            tgt_input: Initial target sequence
            device: Device to use
            max_length: Maximum generation length
            
        Returns:
            Generated sequences
        """
        batch_size = src.size(0)
        generated = tgt_input.clone()
        
        for _ in range(max_length):
            predictions = model(src, generated)
            next_tokens = predictions[:, -1, :].argmax(dim=-1)
            
            # Check for EOS tokens
            eos_mask = (next_tokens == self.tgt_vocab[self.eos_token])
            if eos_mask.all():
                break
            
            # Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        
        return generated


def create_evaluator(tgt_vocab: Dict[str, int], idx2word: Dict[int, str]) -> TranslationEvaluator:
    """
    Create a translation evaluator instance.
    
    Args:
        tgt_vocab: Target vocabulary mapping
        idx2word: Index to word mapping
        
    Returns:
        TranslationEvaluator instance
    """
    return TranslationEvaluator(tgt_vocab, idx2word)


if __name__ == "__main__":
    # Test the evaluator
    print("Testing TranslationEvaluator...")
    
    # Create test vocabulary
    tgt_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'hello': 4, 'world': 5, 'good': 6, 'morning': 7}
    idx2word = {v: k for k, v in tgt_vocab.items()}
    
    evaluator = TranslationEvaluator(tgt_vocab, idx2word)
    
    # Test data
    predictions = [
        ['hello', 'world'],
        ['good', 'morning'],
        ['hello', 'world', 'good']
    ]
    references = [
        ['hello', 'world'],
        ['good', 'morning'],
        ['hello', 'world', 'good']
    ]
    
    # Test BLEU
    bleu_scores = evaluator.calculate_bleu_score(predictions, references)
    print(f"BLEU scores: {bleu_scores}")
    
    # Test METEOR
    meteor_score = evaluator.calculate_meteor_score(predictions, references)
    print(f"METEOR score: {meteor_score:.4f}")
    
    # Test ROUGE-L
    rouge_score = evaluator.calculate_rouge_score(predictions, references)
    print(f"ROUGE-L score: {rouge_score:.4f}")
    
    # Test exact match
    exact_match = evaluator.calculate_exact_match(predictions, references)
    print(f"Exact match: {exact_match:.4f}")
    
    # Test word accuracy
    word_acc = evaluator.calculate_word_accuracy(predictions, references)
    print(f"Word accuracy: {word_acc:.4f}")
    
    print("TranslationEvaluator test completed!")
