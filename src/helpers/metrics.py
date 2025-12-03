import torch
import numpy as np
from collections import Counter
from nltk.translate.meteor_score import meteor_score as nltk_meteor
import warnings

warnings.filterwarnings('ignore')


# BLEU measures n-gram precision with brevity penalty
# Higher BLEU = better overlap between generated and reference captions

def get_ngrams(tokens, n):
    # Extract n-grams from a list of tokens
    # Example: ['the', 'cat', 'sits'] with n=2 -> [('the', 'cat'), ('cat', 'sits')]
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams


def compute_bleu_score(reference, candidate, max_n=4):
    # Calculate BLEU score for a single reference-candidate pair
    # reference: ground truth caption (string or list of words)
    # candidate: generated caption (string or list of words)
    # max_n: maximum n-gram order (4 for BLEU-4)
    
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
        # Example: if candidate has "the the the" and reference has "the cat"
        # only count 1 match for "the", not 3
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
    # Calculate ROUGE-1, ROUGE-2, ROUGE-L
    # ROUGE-1: unigram recall
    # ROUGE-2: bigram recall
    # ROUGE-L: longest common subsequence F1
    
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


# METEOR considers:
# - Exact matches
# - Stemming (e.g., "running" vs "run")
# - Synonyms (e.g., "happy" vs "joyful")
# - Word order (via alignment)

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


# ============================================================================
# EMOTION CLASSIFICATION METRICS (For PowerfulCNN's Aux Head)
# ============================================================================

def compute_emotion_accuracy(logits, targets):
    # Simple accuracy for emotion classification
    # logits: (batch, num_emotions) raw scores
    # targets: (batch,) emotion labels
    
    predictions = torch.argmax(logits, dim=-1)
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


def evaluate_model(references, candidates, verbose=True):
    # Compute all metrics at once
    # references: list of ground truth captions
    # candidates: list of generated captions
    
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
            print("Warning: METEOR calculation failed (NLTK missing?)")
    
    if verbose:
        print("="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        for metric, score in metrics.items():
            print(f"{metric:15s}: {score:.4f}")
        print("="*70)
    
    return metrics



if __name__ == "__main__":
    print("="*70)
    print("TESTING METRICS MODULE")
    print("="*70)
    
    # Test data
    refs = [
        "this painting evokes a feeling of melancholy and isolation",
        "the vibrant colors create a sense of joy and excitement",
        "dark tones suggest sadness and despair"
    ]
    
    cands_good = [
        "this artwork expresses melancholy and feelings of isolation",
        "bright colors evoke joy and excitement",
        "dark colors suggest sadness and despair"
    ]
    
    cands_bad = [
        "this is a painting",
        "there are colors",
        "it is dark"
    ]
    
    print("\n[TEST 1] Good predictions:")
    print("-" * 70)
    metrics_good = evaluate_model(refs, cands_good, verbose=True)
    
    print("\n[TEST 2] Bad predictions:")
    print("-" * 70)
    metrics_bad = evaluate_model(refs, cands_bad, verbose=True)
    
    print("\n[TEST 3] Word accuracy (PyTorch tensors):")
    print("-" * 70)
    # Simulated batch
    predictions = torch.randn(2, 10, 5000)  # (batch=2, seq=10, vocab=5000)
    targets = torch.randint(0, 5000, (2, 10))  # (batch=2, seq=10)
    predictions = predictions.scatter(2, targets.unsqueeze(-1), 10.0)  # Make targets have highest logits
    
    acc = compute_word_accuracy(predictions, targets, pad_idx=0)
    print(f"Word Accuracy: {acc:.4f} (should be ~1.0 for this simulated case)")
    
    print("\n[TEST 4] Emotion metrics:")
    print("-" * 70)
    emotion_logits = torch.randn(32, 9)  # (batch=32, num_emotions=9)
    emotion_targets = torch.randint(0, 9, (32,))  # (batch=32)
    
    emo_acc = compute_emotion_accuracy(emotion_logits, emotion_targets)
    emo_f1 = compute_emotion_f1(emotion_logits, emotion_targets, num_classes=9)
    print(f"Emotion Accuracy: {emo_acc:.4f}")
    print(f"Emotion F1 (macro): {emo_f1:.4f}")
    
    print("\n[TEST 5] METEOR score:")
    print("-" * 70)
    ref_meteor = "this dark painting evokes feelings of sadness"
    cand_meteor = "this artwork shows sadness and melancholy"
    meteor_score = compute_meteor(ref_meteor, cand_meteor)
    print(f"METEOR: {meteor_score:.4f}")
    
    print("\n" + "="*70)
    print("[SUCCESS] All metrics working correctly!")
    print("="*70)

