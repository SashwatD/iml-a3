import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def get_tfidf_embeddings(vocab_list, captions, embedding_dim=256):
    """
    Generates TF-IDF embeddings reduced to embedding_dim using LSA (SVD).
    
    Args:
        vocab_list: List of words in the vocabulary (itos list).
        captions: List of string captions.
        embedding_dim: Target dimension for the embeddings.
        
    Returns:
        embedding_matrix: A numpy array of shape (vocab_size, embedding_dim).
    """
    print("Generating TF-IDF embeddings...")
    
    if captions is None:
        raise ValueError("Captions list is required for TF-IDF embeddings.")
        
    return compute_tfidf_matrix(captions, vocab_list, embedding_dim=embedding_dim)

def compute_tfidf_matrix(captions, vocab_list, embedding_dim=256):
    """
    Computes TF-IDF matrix for the vocabulary using the provided captions and reduces dimensionality via SVD.
    """
    print(f"Computing TF-IDF on {len(captions)} captions...")
    
    vocab_dict = {word: i for i, word in enumerate(vocab_list)}
    
    tfidf = TfidfVectorizer(vocabulary=vocab_dict, token_pattern=r"(?u)\b\w+\b")
    try:
        tfidf_matrix = tfidf.fit_transform(captions) # (n_samples, n_features)
    except ValueError:
        # Fallback if vocab is empty or issues
        print("Error in TF-IDF fit_transform. Returning random embeddings.")
        return np.random.normal(scale=0.1, size=(len(vocab_list), embedding_dim)).astype("float32")
    
    # Reduce dimensionality with SVD (LSA)
    print(f"Reducing dimension to {embedding_dim} using SVD...")
    svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
    word_features = svd.fit_transform(tfidf_matrix.T)
    
    # word_features is (n_features, embedding_dim) which corresponds to (vocab_size, embedding_dim)
    # providing the vocab was strictly followed.
    
    return word_features.astype("float32")
