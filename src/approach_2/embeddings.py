import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os

def get_tfidf_embeddings(vocab_list, captions, embedding_dim=256):
    print("Generating TF-IDF embeddings...")
    
    if captions is None:
        raise ValueError("Captions list is required for TF-IDF embeddings.")
        
    return compute_tfidf_matrix(captions, vocab_list, embedding_dim=embedding_dim)

def compute_tfidf_matrix(captions, vocab_list, embedding_dim=256):
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
    
    return word_features.astype("float32")

def load_binary(fname):    
    embeddings = {}
    
    with open(fname, 'rb') as f:
        header = f.readline()
        try:
            vocab_size, vector_size = map(int, header.split())
        except ValueError:
            raise ValueError(f"Invalid header in {fname}: {header}")

        binary_len = np.dtype('float32').itemsize * vector_size
        
        for _ in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    break
                if ch != b'\n':
                    word.append(ch)
            
            word_str = b''.join(word).decode('utf-8', errors='ignore')
            
            vector_bytes = f.read(binary_len)
            if len(vector_bytes) != binary_len:
                break # Should not happen if file is complete
                
            vector = np.frombuffer(vector_bytes, dtype='float32')
            embeddings[word_str] = vector
            
    return embeddings, vector_size

def get_pretrained_embeddings(vocab_list, model_name="glove-wiki-gigaword-100", embedding_dim=100, captions=None):
    # Check local downloads first
    local_path = os.path.join('./downloads/embeddings', f"{model_name}.bin")
    
    embeddings_dict = {}
    model_dim = 0
    
    print(f"Loading pre-trained model: {model_name}...")
    if os.path.exists(local_path):
        print(f"Found local embedding file: {local_path}")
        embeddings_dict, model_dim = load_binary(local_path)
    elif os.path.exists(model_name):
        embeddings_dict, model_dim = load_binary(model_name)
    else:
        raise FileNotFoundError(f"Could not find embedding file at {local_path} or {model_name}. Please run src/utils/download_embeddings.py first.")

    vocab_size = len(vocab_list)
    
    # Check dimension
    if model_dim != embedding_dim:
        print(f"Using model dim {model_dim}.")
        embedding_dim = model_dim
        
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    hits = 0
    misses = 0
    
    for i, word in enumerate(vocab_list):
        if word in embeddings_dict:
            embedding_matrix[i] = embeddings_dict[word]
            hits += 1
        else:
            # Initialize OOV with random normal
            if i == 0: # <pad>
                continue 
            embedding_matrix[i] = np.random.normal(scale=0.1, size=embedding_dim)
            misses += 1
            
    print(f"Embedding matrix created. Shape: {embedding_matrix.shape}")
    print(f"Hits: {hits}, Misses: {misses}, Coverage: {hits/vocab_size:.2%}")
    
    return embedding_matrix
