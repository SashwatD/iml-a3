import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
import string
import re
import pickle

def load_and_clean_data(csv_path, image_dir, sample_size=None, stratify_col='art_style'):
    """
    Loads dataset, filters missing images, and performs stratified sampling.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
        
    # Let's verify column names first if possible, but standard ArtEmis has 'art_style', 'painting'
    if 'image_file' not in df.columns:
         # Construct path if not present
         df['image_file'] = df.apply(lambda x: os.path.join(image_dir, x['art_style'], x['painting'] + '.jpg'), axis=1)
    else:
         # If absolute or relative paths are already there, ensure they are correct
         pass

    # Filter missing files
    print("Checking for missing files...")
    def file_exists(path):
        return os.path.exists(path)
    
    # This might be slow for 80k images, but necessary for robustness
    valid_mask = df['image_file'].apply(file_exists)
    missing_count = (~valid_mask).sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} images not found. Removing them.")
        df = df[valid_mask]
    
    # Stratified Sampling
    if sample_size and sample_size < len(df):
        print(f"Performing stratified sampling to reduce size to {sample_size}...")
        # Use train_test_split for stratified sampling
        try:
            df, _ = train_test_split(
                df, 
                train_size=sample_size, 
                stratify=df[stratify_col], 
                random_state=42
            )
        except ValueError as e:
            print(f"Stratified sampling failed (likely due to rare classes): {e}")
            print("Falling back to random sampling.")
            df = df.sample(n=sample_size, random_state=42)
            
    print(f"Final dataset size: {len(df)}")
    return df

def custom_standardization(input_string):
    """
    Custom text standardization: lowercase, remove punctuation.
    """
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(string.punctuation)}]", "")

def save_vectorizer(vectorizer, filepath):
    """Saves the TextVectorization configuration and vocabulary."""
    config = vectorizer.get_config()
    
    # If 'standardize' is a function (callable), we can't pickle it easily in the config for from_config
    # We will remove it from the config to be saved, and re-attach it on load.
    if callable(config.get('standardize')):
        config['standardize'] = None
        
    data = {
        'config': config,
        'vocabulary': vectorizer.get_vocabulary(),
        'weights': vectorizer.get_weights() # Keep weights just in case, but vocab is key
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Vectorizer saved to {filepath}")

def load_vectorizer_from_disk(filepath):
    """Loads a TextVectorization layer from disk."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    config = data['config']
    vectorizer = TextVectorization.from_config(config)
    
    # Re-attach custom standardization
    vectorizer.standardize = custom_standardization
    
    # Set vocabulary explicitly
    if 'vocabulary' in data:
        vectorizer.set_vocabulary(data['vocabulary'])
    else:
        # Fallback to weights if vocab not present (legacy support if needed)
        vectorizer.set_weights(data['weights'])
        
    print(f"Vectorizer loaded from {filepath}")
    return vectorizer

def get_augmentation_layer():
    """Returns a Sequential model for image augmentation."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

def create_tf_dataset(
    df, 
    image_size=(256, 256), 
    batch_size=32, 
    vocab_size=5000, 
    max_length=50, 
    validation_split=0.2,
    augment=False,
    vectorizer=None
):
    """
    Creates tf.data.Dataset pipeline.
    """
    
    # Prepare paths and captions
    image_paths = df['image_file'].values
    captions = df['utterance'].values # Assuming 'utterance' is the caption column in ArtEmis
    
    # Add start and end tokens
    captions = [f"<start> {cap} <end>" for cap in captions]
    
    # Split data
    train_paths, val_paths, train_caps, val_caps = train_test_split(
        image_paths, captions, test_size=validation_split, random_state=42
    )
    
    # Text Vectorization
    if vectorizer is None:
        vectorizer = TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_length,
            standardize=custom_standardization
        )
        # Adapt vectorizer on training data
        print("Adapting text vectorizer...")
        vectorizer.adapt(train_caps)
    else:
        print("Using provided vectorizer.")
    
    def read_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.image.convert_image_dtype(img, tf.float32) # Normalize to [0, 1]
        return img
    
    # Augmentation layer
    aug_layer = get_augmentation_layer() if augment else None
    
    def process_data(image_path, caption, training=False):
        img = read_image(image_path)
        
        if training and aug_layer is not None:
            img = aug_layer(img)
            
        cap = vectorizer(caption)
        
        # Split into input and target for teacher forcing
        # Input: <start> w1 w2 ... wn
        # Target: w1 w2 ... wn <end>
        
        cap_in = cap[:-1]
        cap_out = cap[1:]
        
        return (img, cap_in), cap_out
    
    def make_dataset(paths, caps, is_training=False):
        dataset = tf.data.Dataset.from_tensor_slices((paths, caps))
        # Use lambda to pass training flag
        dataset = dataset.map(
            lambda p, c: process_data(p, c, training=is_training), 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    print("Creating training dataset...")
    train_ds = make_dataset(train_paths, train_caps, is_training=augment)
    
    print("Creating validation dataset...")
    val_ds = make_dataset(val_paths, val_caps, is_training=False)
    
    return train_ds, val_ds, vectorizer

if __name__ == "__main__":
    # Test the pipeline
    # Note: Adjust paths as per your local setup for testing
    CSV_PATH = "data/images/artemis_dataset_release_v0.csv"
    IMG_DIR = "data/images/wikiart"
    
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR):
        df = load_and_clean_data(CSV_PATH, IMG_DIR, sample_size=100)
        train_ds, val_ds, vectorizer = create_tf_dataset(df, batch_size=4, augment=True)
        
        # Test saving/loading
        save_vectorizer(vectorizer, "test_vectorizer.pkl")
        loaded_vec = load_vectorizer_from_disk("test_vectorizer.pkl")
        
        for img, cap in train_ds.take(1):
            print(f"Image batch shape: {img.shape}")
            print(f"Caption batch shape: {cap.shape}")
            print("Vocab size:", len(vectorizer.get_vocabulary()))
            
        # Clean up
        if os.path.exists("test_vectorizer.pkl"):
            os.remove("test_vectorizer.pkl")
    else:
        print("Dataset not found at default paths. Skipping test run.")
