import os
import tensorflow as tf
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.layers import TextVectorization
import pickle

from src.preprocessing.preprocessing import load_and_clean_data, custom_standardization
from src.approach_2.caption_model import build_vit_caption_model

def load_vectorizer(vectorizer_config_path=None, df=None, vocab_size=5000, max_length=50):
    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_length,
        standardize=custom_standardization
    )
    
    if df is not None:
        captions = [f"<start> {cap} <end>" for cap in df['utterance'].values]
        vectorizer.adapt(captions)
        
    return vectorizer

def generate_caption(model, image, vectorizer, max_length=50):
    # Image: (256, 256, 3) -> (1, 256, 256, 3)
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)
        
    vocab = vectorizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    
    # Start token
    start_token = vectorizer(["<start>"])[:, 0] # get index of <start>
    
    # Let's check the index of "start" (standardized)
    try:
        start_index = list(vocab).index("start")
        end_index = list(vocab).index("end")
    except ValueError:
        # Fallback if tokens stripped or different
        print("Warning: 'start' or 'end' token not found in vocab. Using indices 2 and 3.")
        start_index = 2
        end_index = 3

    output = tf.expand_dims([start_index], 0)
    
    for i in range(max_length - 1):
        # Pad to max_length for the model input if needed, or just pass current sequence
        # The model expects (batch, max_length) usually for PositionalEmbedding
        
        # Current sequence length
        curr_len = tf.shape(output)[1]
        
        # Pad to max_length
        padded_seq = tf.pad(output, [[0, 0], [0, max_length - curr_len]])
        
        predictions = model.predict([image, padded_seq], verbose=0)
        
        predictions = predictions[:, i, :] # Get logits for the next token
        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)
        
        if predicted_id == end_index:
            break
            
        output = tf.concat([output, predicted_id], axis=-1)
        
    # Convert to text
    output_ids = output.numpy()[0]
    caption = " ".join([index_lookup[idx] for idx in output_ids if idx not in [0, start_index, end_index]])
    return caption

def evaluate_model(model, df, image_dir, vectorizer, sample_size=100):
    print(f"Evaluating on {sample_size} samples...")
    
    actuals = []
    predictions = []
    
    # Sample data
    sample_df = df.sample(n=sample_size, random_state=42)
    
    for idx, row in sample_df.iterrows():
        image_path = row['image_file']
        true_caption = row['utterance']
        
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (256, 256))
            img = tf.image.convert_image_dtype(img, tf.float32)
            
            pred_caption = generate_caption(model, img, vectorizer)
            
            actuals.append([true_caption.split()])
            predictions.append(pred_caption.split())
            
            if idx % 20 == 0:
                print(f"Img: {row['painting']}")
                print(f"True: {true_caption}")
                print(f"Pred: {pred_caption}")
                print("-" * 30)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    # Calculate BLEU
    bleu1 = corpus_bleu(actuals, predictions, weights=(1.0, 0, 0, 0))
    bleu4 = corpus_bleu(actuals, predictions, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    return bleu1, bleu4

if __name__ == "__main__":
    CSV_PATH = "data/images/artemis_dataset_release_v0.csv"
    IMG_DIR = "data/images/wikiart"
    MODEL_WEIGHTS = "models/approach-2/final_model.weights.h5"
    
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR) and os.path.exists(MODEL_WEIGHTS):
        # Load data for vectorizer adaptation
        df = load_and_clean_data(CSV_PATH, IMG_DIR, sample_size=1000)
        vectorizer = load_vectorizer(df=df)
        
        # Build model
        model = build_vit_caption_model(
            input_shape=(256, 256, 3),
            vocab_size=5000,
            max_length=50
        )
        model.load_weights(MODEL_WEIGHTS)
        
        evaluate_model(model, df, IMG_DIR, vectorizer)
    else:
        print("Files not found. Skipping evaluation.")
