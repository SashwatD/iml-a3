import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize

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

def get_pretrained_embeddings(vocab_list, model_name="glove-wiki-gigaword-100", embedding_dim=100, captions=None):
    # Ensure nltk resources are available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

    model = None
    
    if model_name == "word2vec":
        print("Training Word2Vec from scratch...")x
        if captions is None:
            raise ValueError("Captions are required for training Word2Vec.")
            
        # Tokenize captions
        print("Tokenizing captions for Word2Vec...")
        tokenized_captions = [word_tokenize(cap.lower()) for cap in captions]
        
        print("Training Word2Vec model...")
        model = Word2Vec(sentences=tokenized_captions, vector_size=embedding_dim, window=5, min_count=1, workers=4)
        print("Word2Vec training complete.")
        
        # Word2Vec model.wv is the KeyedVectors instance
        wv = model.wv
        
    else:
        print(f"Loading pre-trained model: {model_name}...")
        try:
            # Check if it's a path or a gensim-data model
            if os.path.exists(model_name):
                # Load from file (assuming KeyedVectors format or similar)
                from gensim.models import KeyedVectors
                wv = KeyedVectors.load_word2vec_format(model_name, binary=True)
            else:
                # Download/Load from gensim-data
                wv = api.load(model_name)
                
        except Exception as e:
            print(f"Error loading gensim model: {e}")
            print("Ensure you have internet connection to download embeddings.")
            raise e
        
    vocab_size = len(vocab_list)
    
    # Check dimension
    if wv.vector_size != embedding_dim:
        print(f"Warning: Requested embedding_dim {embedding_dim} does not match model dim {wv.vector_size}.")
        print(f"Using model dim {wv.vector_size}.")
        embedding_dim = wv.vector_size
        
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    hits = 0
    misses = 0
    
    for i, word in enumerate(vocab_list):
        if word in wv:
            embedding_matrix[i] = wv[word]
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
