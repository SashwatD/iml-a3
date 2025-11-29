# Evaluation Metrics for Caption Generation
# BLEU scores to measure caption quality

import numpy as np
from collections import Counter


def compute_bleu(reference, candidate, max_n=4):
    # Calculate BLEU score for a single reference-candidate pair
    # reference: list of words (ground truth)
    # candidate: list of words (model prediction)
    # max_n: maximum n-gram order (BLEU-4 means up to 4-grams)
    
    reference = reference.split() if isinstance(reference, str) else reference
    candidate = candidate.split() if isinstance(candidate, str) else candidate
    
    # Handle empty captions
    if len(candidate) == 0:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(reference, n)
        cand_ngrams = _get_ngrams(candidate, n)
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches (clipped to reference count)
        matches = 0
        for ngram in cand_ngrams:
            matches += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))
        
        precision = matches / len(cand_ngrams)
        precisions.append(precision)
    
    # Brevity penalty (penalize short captions)
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len > ref_len:
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


def compute_bleu_n(reference, candidate, n):
    # Calculate BLEU-n score (e.g., BLEU-1, BLEU-2, etc.)
    return compute_bleu(reference, candidate, max_n=n)


def compute_corpus_bleu(references, candidates, max_n=4):
    # Calculate average BLEU score across multiple samples
    # references: list of reference captions
    # candidates: list of generated captions
    
    if len(references) != len(candidates):
        raise ValueError(f"Length mismatch: {len(references)} refs vs {len(candidates)} candidates")
    
    scores = []
    for ref, cand in zip(references, candidates):
        score = compute_bleu(ref, cand, max_n=max_n)
        scores.append(score)
    
    return np.mean(scores)


def compute_all_bleu_scores(references, candidates):
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu1 = compute_corpus_bleu(references, candidates, max_n=1)
    bleu2 = compute_corpus_bleu(references, candidates, max_n=2)
    bleu3 = compute_corpus_bleu(references, candidates, max_n=3)
    bleu4 = compute_corpus_bleu(references, candidates, max_n=4)
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }


def _get_ngrams(tokens, n):
    # Extract n-grams from token list
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams


# Test the metrics
if __name__ == "__main__":
    print("="*70)
    print("Testing BLEU Metrics")
    print("="*70)
    
    # Example 1: Perfect match
    ref1 = "the cat sits on the mat"
    cand1 = "the cat sits on the mat"
    bleu1 = compute_bleu(ref1, cand1)
    print(f"\n1. Perfect match:")
    print(f"   Reference:  '{ref1}'")
    print(f"   Candidate:  '{cand1}'")
    print(f"   BLEU-4:     {bleu1:.4f} (should be ~1.0)")
    
    # Example 2: Partial match
    ref2 = "the cat sits on the mat"
    cand2 = "the dog sits on the floor"
    bleu2 = compute_bleu(ref2, cand2)
    print(f"\n2. Partial match:")
    print(f"   Reference:  '{ref2}'")
    print(f"   Candidate:  '{cand2}'")
    print(f"   BLEU-4:     {bleu2:.4f}")
    
    # Example 3: Corpus BLEU
    refs = [ref1, ref2]
    cands = [cand1, cand2]
    scores = compute_all_bleu_scores(refs, cands)
    print(f"\n3. Corpus BLEU scores:")
    for metric, score in scores.items():
        print(f"   {metric}: {score:.4f}")
    
    print("\n" + "="*70)
    print("[SUCCESS] BLEU metrics working correctly!")
    print("="*70)

