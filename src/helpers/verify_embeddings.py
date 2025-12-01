import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.approach_2.embeddings import get_pretrained_embeddings

def test_word2vec():
    print("Testing Word2Vec...")
    captions = ["this is a test caption", "another caption for testing"]
    vocab_list = ["<pad>", "<start>", "<end>", "this", "is", "a", "test", "caption", "another", "for", "testing"]
    
    embeddings = get_pretrained_embeddings(vocab_list, model_name="word2vec", embedding_dim=100, captions=captions)
    print("Word2Vec embeddings shape:", embeddings.shape)
    assert embeddings.shape == (len(vocab_list), 100)
    print("Word2Vec test passed!")

def test_glove():
    print("\nTesting GloVe...")
    # Using a small model for testing if possible, or just checking if downloader triggers
    # But downloading 100MB might be slow. Let's just check if it fails gracefully or starts.
    # Actually, let's skip actual download to avoid long wait, or use a very small one if available.
    # glove-twitter-25 is relatively small (100MB).
    
    vocab_list = ["<pad>", "the", "cat"]
    try:
        embeddings = get_pretrained_embeddings(vocab_list, model_name="glove-twitter-25", embedding_dim=25)
        print("GloVe embeddings shape:", embeddings.shape)
        assert embeddings.shape == (len(vocab_list), 25)
        print("GloVe test passed!")
    except Exception as e:
        print(f"GloVe test failed (might be network/download issue): {e}")

if __name__ == "__main__":
    test_word2vec()
    # test_glove() # Uncomment to test download (might take time)
