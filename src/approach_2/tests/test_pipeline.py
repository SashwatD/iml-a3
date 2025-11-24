import os
import tensorflow as tf
import numpy as np
from src.preprocessing.preprocessing import load_and_clean_data, create_tf_dataset, save_vectorizer, load_vectorizer_from_disk
from src.approach_2.caption_model import build_vit_caption_model

def test_pipeline_and_model():
    print("Testing Data Pipeline...")
    CSV_PATH = "data/images/artemis_dataset_release_v0.csv"
    IMG_DIR = "data/images/wikiart"
    
    if not os.path.exists(CSV_PATH) or not os.path.exists(IMG_DIR):
        print("Dataset not found. Skipping test.")
        return

    # 1. Test Data Loading & Augmentation
    df = load_and_clean_data(CSV_PATH, IMG_DIR, sample_size=50)
    train_ds, val_ds, vectorizer = create_tf_dataset(df, batch_size=4, augment=True)
    
    # Check batch shapes
    for (img, cap_in), cap_out in train_ds.take(1):
        print(f"Train Image Batch: {img.shape}")
        print(f"Train Caption Input Batch: {cap_in.shape}")
        print(f"Train Caption Output Batch: {cap_out.shape}")
        assert img.shape == (4, 256, 256, 3)
        
    # 2. Test Vectorizer Persistence
    print("Testing Vectorizer Persistence...")
    save_vectorizer(vectorizer, "test_vec.pkl")
    loaded_vec = load_vectorizer_from_disk("test_vec.pkl")
    
    vocab_orig = vectorizer.get_vocabulary()
    vocab_loaded = loaded_vec.get_vocabulary()
    
    print(f"Original Vocab Size: {len(vocab_orig)}")
    print(f"Loaded Vocab Size: {len(vocab_loaded)}")
    if len(vocab_orig) > 0:
        print(f"Orig Head: {vocab_orig[:5]}")
        print(f"Loaded Head: {vocab_loaded[:5]}")
        
    assert vocab_orig == vocab_loaded
    print("Vectorizer saved and loaded correctly.")
    
    if os.path.exists("test_vec.pkl"):
        os.remove("test_vec.pkl")

    # 3. Test Model Build & Forward Pass
    print("Testing Model Build...")
    model = build_vit_caption_model(
        input_shape=(256, 256, 3),
        vocab_size=len(vocab_orig),
        max_length=49 # 50 - 1
    )
    
    # Forward pass with dummy data
    dummy_img = tf.random.normal((2, 256, 256, 3))
    dummy_cap = tf.random.uniform((2, 49), minval=0, maxval=100, dtype=tf.int64)
    
    output = model([dummy_img, dummy_cap])
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 49, len(vocab_orig))
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline_and_model()
