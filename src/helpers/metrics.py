import torch
import numpy as np
from collections import Counter
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import warnings

warnings.filterwarnings('ignore')

import nltk
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def get_ngrams(tokens, n):
    # Extract n-grams from a list of tokens
    # Example: ['the', 'cat', 'sits'] with n=2 -> [('the', 'cat'), ('cat', 'sits')]
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams


def compute_bleu_score(reference, candidate, max_n=4):
    # Calculate BLEU score for a single reference-candidate pair
    
    # Convert strings to token lists
    if isinstance(reference, str):
        reference = reference.lower().split()
    if isinstance(candidate, str):
        candidate = candidate.lower().split()
    
    # Edge case: empty candidate
    if len(candidate) == 0:
        return 0.0
    
    # Calculate precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(get_ngrams(reference, n))
        cand_ngrams = Counter(get_ngrams(candidate, n))
        
        # No n-grams of this order
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count clipped matches
        matches = 0
        for ngram in cand_ngrams:
            matches += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))
        
        precision = matches / sum(cand_ngrams.values())
        precisions.append(precision)
    
    # Brevity penalty (penalize too-short captions)
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        log_precisions = [np.log(p) for p in precisions]
        geo_mean = np.exp(sum(log_precisions) / len(precisions))
    else:
        geo_mean = 0.0
    
    bleu = bp * geo_mean
    return bleu


def compute_corpus_bleu(references, candidates, max_n=4):
    # Average BLEU score across multiple samples
    scores = [compute_bleu_score(ref, cand, max_n) 
              for ref, cand in zip(references, candidates)]
    return np.mean(scores)


def compute_all_bleu_scores(references, candidates):
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    # BLEU-1: unigram precision (individual words)
    # BLEU-2: bigram precision (word pairs)
    # BLEU-3: trigram precision
    # BLEU-4: 4-gram precision (standard metric)
    
    return {
        'BLEU-1': compute_corpus_bleu(references, candidates, max_n=1),
        'BLEU-2': compute_corpus_bleu(references, candidates, max_n=2),
        'BLEU-3': compute_corpus_bleu(references, candidates, max_n=3),
        'BLEU-4': compute_corpus_bleu(references, candidates, max_n=4),
    }


# ROUGE measures recall (how much of reference is captured)
# Unlike BLEU (precision-focused), ROUGE is recall-focused

def compute_rouge_n(reference, candidate, n=1):
    # ROUGE-N: n-gram recall
    # Measures: what fraction of reference n-grams appear in candidate
    
    if isinstance(reference, str):
        reference = reference.lower().split()
    if isinstance(candidate, str):
        candidate = candidate.lower().split()
    
    if len(reference) == 0:
        return 0.0
    
    ref_ngrams = Counter(get_ngrams(reference, n))
    cand_ngrams = Counter(get_ngrams(candidate, n))
    
    if len(ref_ngrams) == 0:
        return 0.0
    
    # Count overlapping n-grams
    matches = 0
    for ngram in ref_ngrams:
        matches += min(ref_ngrams[ngram], cand_ngrams.get(ngram, 0))
    
    # Recall = matches / total reference n-grams
    recall = matches / sum(ref_ngrams.values())
    return recall


def compute_rouge_l(reference, candidate):
    # ROUGE-L: Longest Common Subsequence (LCS)
    # Captures sentence-level similarity
    # Example: ref="A B C D", cand="A C D E" -> LCS="A C D" (length 3)
    
    if isinstance(reference, str):
        reference = reference.lower().split()
    if isinstance(candidate, str):
        candidate = candidate.lower().split()
    
    # Dynamic programming to find LCS length
    m, n = len(reference), len(candidate)
    if m == 0 or n == 0:
        return 0.0
    
    # DP table
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == candidate[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
    
    lcs_length = lcs[m][n]
    
    # F1-score based on LCS
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / n
    recall = lcs_length / m
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_all_rouge_scores(references, candidates):
    rouge1_scores = [compute_rouge_n(ref, cand, n=1) 
                     for ref, cand in zip(references, candidates)]
    rouge2_scores = [compute_rouge_n(ref, cand, n=2) 
                     for ref, cand in zip(references, candidates)]
    rougeL_scores = [compute_rouge_l(ref, cand) 
                     for ref, cand in zip(references, candidates)]
    
    return {
        'ROUGE-1': np.mean(rouge1_scores),
        'ROUGE-2': np.mean(rouge2_scores),
        'ROUGE-L': np.mean(rougeL_scores),
    }

def compute_meteor(reference, candidate):
    # METEOR: Metric for Evaluation of Translation with Explicit ORdering
    # More sophisticated than BLEU, accounts for semantic similarity
    
    try:
        if isinstance(reference, str):
            reference = reference.lower().split()
        if isinstance(candidate, str):
            candidate = candidate.lower().split()
        
        # NLTK's METEOR implementation
        # requires reference as list of tokens
        score = nltk_meteor([reference], candidate)
        return score
    except Exception as e:
        # Fallback if NLTK not available or error
        print(f"Warning: METEOR failed ({e}), returning 0.0")
        return 0.0


def compute_corpus_meteor(references, candidates):
    # Average METEOR score across multiple samples
    scores = [compute_meteor(ref, cand) 
              for ref, cand in zip(references, candidates)]
    return np.mean(scores)


# Simple metric: fraction of correctly predicted words

def compute_word_accuracy(predictions, targets, pad_idx=0):
    # predictions: (batch, seq_len, vocab_size) logits
    # targets: (batch, seq_len) ground truth indices
    # pad_idx: index of <pad> token to ignore
    
    # Get predicted indices
    pred_indices = torch.argmax(predictions, dim=-1)  # (batch, seq_len)
    
    # Create mask (ignore padding)
    mask = (targets != pad_idx).float()
    
    # Count correct predictions
    correct = ((pred_indices == targets).float() * mask).sum()
    total = mask.sum()
    
    if total == 0:
        return 0.0
    
    accuracy = (correct / total).item()
    return accuracy

def compute_emotion_accuracy(logits, targets):
    # Simple accuracy for emotion classification
    # logits: (batch, num_emotions) raw scores
    # targets: (batch,) emotion labels
    
    predictions = torch.argmax(logits, dim=-1).cpu()
    targets = targets.cpu()
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_emotion_f1(logits, targets, num_classes=9):
    # F1 score for emotion classification (handles class imbalance)
    
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Per-class precision and recall
    f1_scores = []
    for cls in range(num_classes):
        tp = ((predictions == cls) & (targets == cls)).sum()
        fp = ((predictions == cls) & (targets != cls)).sum()
        fn = ((predictions != cls) & (targets == cls)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    # Macro-averaged F1
    return np.mean(f1_scores)


def evaluate_model(references, candidates, emotion_preds=None, emotion_targets=None, verbose=True):
    # Compute all metrics at once
    # references: list of ground truth captions
    # candidates: list of generated captions
    # emotion_preds: (optional) list/tensor of predicted emotion indices or logits
    # emotion_targets: (optional) list/tensor of true emotion indices
    
    metrics = {}
    
    # BLEU scores
    bleu_scores = compute_all_bleu_scores(references, candidates)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = compute_all_rouge_scores(references, candidates)
    metrics.update(rouge_scores)
    
    # METEOR score (optional)
    try:
        meteor = compute_corpus_meteor(references, candidates)
        metrics['METEOR'] = meteor
    except:
        if verbose:
            print("Warning: METEOR calculation failed")
            
    # Emotion Metrics (if provided)
    if emotion_preds is not None and emotion_targets is not None:
        # Ensure tensors
        if not isinstance(emotion_preds, torch.Tensor):
            emotion_preds = torch.tensor(emotion_preds)
        if not isinstance(emotion_targets, torch.Tensor):
            emotion_targets = torch.tensor(emotion_targets)
            
        # If preds are logits (2D), get indices
        if emotion_preds.dim() > 1:
            acc = compute_emotion_accuracy(emotion_preds, emotion_targets)
        else:
            correct = (emotion_preds == emotion_targets).sum().item()
            total = len(emotion_targets)
            acc = correct / total if total > 0 else 0.0
            
        metrics['Emotion Acc'] = acc
    
    if verbose:
        print("Evaluation Results")
        for metric, score in metrics.items():
            print(f"{metric:15s}: {score:.4f}")
    
    return metrics