import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import gensim.downloader as api
import os
import pickle

def get_tfidf_embeddings(vectorizer, captions, embedding_dim=256):
    """
    Generates TF-IDF embeddings reduced to embedding_dim using LSA (SVD).
    
    Args:
        vectorizer: The adapted TextVectorization layer.
        captions: List of string captions (e.g. from dataframe).
        embedding_dim: Target dimension for the embeddings.
        
    Returns:
        embedding_matrix: A numpy array of shape (vocab_size, embedding_dim).
    """
    print("Generating TF-IDF embeddings...")
    
    if captions is None:
        raise ValueError("Captions list is required for TF-IDF embeddings.")
        
    vocab = vectorizer.get_vocabulary()
    
    # Use the helper function
    return compute_tfidf_matrix(captions, vocab, embedding_dim=embedding_dim)

def compute_tfidf_matrix(captions, vocab, embedding_dim=256):
    """
    Computes TF-IDF matrix for the vocabulary using the provided captions and reduces dimensionality via SVD.
    """
    print(f"Computing TF-IDF on {len(captions)} captions...")
    
    # Filter vocab to remove special tokens for sklearn
    clean_vocab = [w for w in vocab if w not in ['', '[UNK]']]
    
    # Create TfidfVectorizer with the specific vocabulary
    tfidf = TfidfVectorizer(vocabulary=clean_vocab, token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = tfidf.fit_transform(captions) # (n_samples, n_features)
    
    # Reduce dimensionality with SVD (LSA)
    # We transpose to get (n_features, n_samples) so SVD gives us feature embeddings
    print(f"Reducing dimension to {embedding_dim} using SVD...")
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    word_features = svd.fit_transform(tfidf_matrix.T)
    
    # Map back to full vocab indices
    final_matrix = np.zeros((len(vocab), embedding_dim), dtype="float32")
    feature_index_map = {word: i for i, word in enumerate(clean_vocab)}
    
    for i, word in enumerate(vocab):
        if word in feature_index_map:
            final_matrix[i] = word_features[feature_index_map[word]]
        else:
            # Initialize special tokens (pad, unk, start, end) randomly
            final_matrix[i] = np.random.normal(scale=0.1, size=embedding_dim)
            
    return final_matrix

def get_pretrained_embeddings(vectorizer, model_name="glove-wiki-gigaword-100", embedding_dim=100):
    """
    Loads pre-trained embeddings using gensim and maps them to the vectorizer's vocabulary.
    
    Args:
        vectorizer: The adapted TextVectorization layer.
        model_name: Gensim model name (e.g., 'glove-wiki-gigaword-100', 'word2vec-google-news-300').
        embedding_dim: Expected dimension (must match model).
        
    Returns:
        embedding_matrix: A numpy array of shape (vocab_size, embedding_dim).
    """
    print(f"Loading pre-trained model: {model_name}...")
    try:
        model = api.load(model_name)
    except Exception as e:
        print(f"Error loading gensim model: {e}")
        print("Ensure you have internet connection to download embeddings.")
        raise e
        
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    
    # Check dimension
    if model.vector_size != embedding_dim:
        print(f"Warning: Requested embedding_dim {embedding_dim} does not match model dim {model.vector_size}.")
        print(f"Using model dim {model.vector_size}.")
        embedding_dim = model.vector_size
        
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    hits = 0
    misses = 0
    
    for i, word in enumerate(vocab):
        if word in model:
            embedding_matrix[i] = model[word]
            hits += 1
        else:
            # Initialize OOV with random normal or zeros
            # Padding (index 0) should ideally be zero, but 'mask_zero=True' in Embedding layer handles it.
            # If we use 0 for padding index explicitly:
            if i == 0:
                continue # Leave as zeros
            embedding_matrix[i] = np.random.normal(scale=0.1, size=embedding_dim)
            misses += 1
            
    print(f"Embedding matrix created. Shape: {embedding_matrix.shape}")
    print(f"Hits: {hits}, Misses: {misses}, Coverage: {hits/vocab_size:.2%}")
    
    return embedding_matrix
