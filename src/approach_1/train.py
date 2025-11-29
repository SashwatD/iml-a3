# Training Script for CNN+LSTM Caption Model
# Trains model with streaming data loader and multiple embedding options

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import argparse
import json
from datetime import datetime

from src.approach_1.caption_model import build_caption_model, compile_model
from src.approach_1.data_loader import load_generators
from src.approach_2.embeddings import get_tfidf_embeddings, get_pretrained_embeddings


def create_callbacks(model_dir, model_name):
    # Set up training callbacks for monitoring and saving progress
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            filepath=str(model_dir / f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Stop training if validation loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Multiply LR by 0.5
            patience=3,  # Wait 3 epochs before reducing
            min_lr=1e-7,
            verbose=1
        ),
        
        # Log metrics to CSV for plotting
        CSVLogger(
            filename=str(model_dir / f'{model_name}_training_log.csv'),
            append=True
        )
    ]
    
    return callbacks


def prepare_embedding_matrix(tokenizer, embedding_type='learned', embedding_dim=256):
    # Create embedding matrix based on chosen strategy
    # Returns None for learned embeddings (trainable from scratch)
    
    vocab_size = len(tokenizer.word_to_index)
    
    if embedding_type == 'learned':
        # No pre-initialization, embeddings learn from scratch
        print(f"Using learned embeddings (trainable, {embedding_dim}D)")
        return None
    
    elif embedding_type == 'tfidf':
        # TF-IDF with dimensionality reduction via SVD
        print(f"Using TF-IDF embeddings (non-trainable, {embedding_dim}D)")
        
        # Need raw captions to compute TF-IDF
        # We'll load them from the processed metadata
        import pickle
        data_path = Path('data/processed/train_data_metadata.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        raw_captions = data['raw_captions']
        
        # Get vocabulary from tokenizer
        vocab = list(tokenizer.word_to_index.keys())
        
        # Compute TF-IDF embeddings
        from src.approach_2.embeddings import compute_tfidf_matrix
        embedding_matrix = compute_tfidf_matrix(raw_captions, vocab, embedding_dim=embedding_dim)
        
        return embedding_matrix
    
    elif embedding_type == 'glove':
        # GloVe pre-trained embeddings
        print("Using GloVe embeddings (non-trainable, 100D)")
        
        # GloVe comes in fixed dimensions, most common is 100D
        embedding_dim = 100
        
        # Create a mock vectorizer-like object for compatibility
        class MockVectorizer:
            def __init__(self, word_to_index):
                self.word_to_index = word_to_index
            def get_vocabulary(self):
                return list(self.word_to_index.keys())
        
        mock_vec = MockVectorizer(tokenizer.word_to_index)
        embedding_matrix = get_pretrained_embeddings(
            mock_vec, 
            model_name='glove-wiki-gigaword-100',
            embedding_dim=100
        )
        
        return embedding_matrix, embedding_dim  # Return both matrix and actual dim
    
    elif embedding_type == 'word2vec':
        # Word2Vec pre-trained embeddings
        print("Using Word2Vec embeddings (non-trainable, 300D)")
        
        # Word2Vec comes in 300D
        embedding_dim = 300
        
        class MockVectorizer:
            def __init__(self, word_to_index):
                self.word_to_index = word_to_index
            def get_vocabulary(self):
                return list(self.word_to_index.keys())
        
        mock_vec = MockVectorizer(tokenizer.word_to_index)
        embedding_matrix = get_pretrained_embeddings(
            mock_vec,
            model_name='word2vec-google-news-300',
            embedding_dim=300
        )
        
        return embedding_matrix, embedding_dim
    
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def train_model(
    model_name='cnn_lstm_learned',
    embedding_type='learned',
    batch_size=32,
    epochs=30,
    learning_rate=0.001,
    embedding_dim=256,
    lstm_units=256,
    dropout_rate=0.5,
    image_size=(128, 128),
    output_dir='models/approach_1'
):
    # Main training function
    
    print("="*70)
    print(f"TRAINING: {model_name}")
    print(f"Embedding Type: {embedding_type}")
    print("="*70)
    
    # Step 1: Load data generators
    print("\n1. Loading data...")
    train_gen, val_gen, test_gen, tokenizer = load_generators(
        batch_size=batch_size,
        image_size=image_size
    )
    
    vocab_size = len(tokenizer.word_to_index)
    max_length = tokenizer.max_length
    
    print(f"   Vocabulary size: {vocab_size:,}")
    print(f"   Max caption length: {max_length}")
    print(f"   Train: {len(train_gen.image_paths):,} samples")
    print(f"   Val: {len(val_gen.image_paths):,} samples")
    
    # Step 2: Prepare embeddings
    print("\n2. Preparing embeddings...")
    embedding_result = prepare_embedding_matrix(
        tokenizer, 
        embedding_type=embedding_type,
        embedding_dim=embedding_dim
    )
    
    # Handle different return types
    if embedding_result is None:
        embedding_matrix = None
        final_embedding_dim = embedding_dim
    elif isinstance(embedding_result, tuple):
        embedding_matrix, final_embedding_dim = embedding_result
    else:
        embedding_matrix = embedding_result
        final_embedding_dim = embedding_dim
    
    # Step 3: Build model
    print("\n3. Building model...")
    encoder, decoder, full_model = build_caption_model(
        vocab_size=vocab_size,
        embedding_dim=final_embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length,
        dropout_rate=dropout_rate,
        embedding_matrix=embedding_matrix,
        trainable_embeddings=(embedding_type == 'learned')
    )
    
    print(f"   Total parameters: {full_model.count_params():,}")
    
    # Step 4: Compile model
    print("\n4. Compiling model...")
    compile_model(full_model, learning_rate=learning_rate)
    
    # Step 5: Setup callbacks
    print("\n5. Setting up callbacks...")
    callbacks = create_callbacks(output_dir, model_name)
    
    # Step 6: Save configuration
    config = {
        'model_name': model_name,
        'embedding_type': embedding_type,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'embedding_dim': final_embedding_dim,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'image_size': list(image_size),
        'vocab_size': vocab_size,
        'max_length': max_length,
        'train_samples': len(train_gen.image_paths),
        'val_samples': len(val_gen.image_paths),
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = Path(output_dir) / f'{model_name}_config.json'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Config saved to {config_path}")
    
    # Step 7: Train
    print("\n6. Starting training...")
    print("="*70)
    
    history = full_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 8: Save models
    print("\n7. Saving models...")
    output_dir = Path(output_dir)
    
    final_model_path = output_dir / f'{model_name}_final.h5'
    encoder_path = output_dir / f'{model_name}_encoder.h5'
    decoder_path = output_dir / f'{model_name}_decoder.h5'
    
    full_model.save(final_model_path)
    encoder.save(encoder_path)
    decoder.save(decoder_path)
    
    print(f"   Final model: {final_model_path}")
    print(f"   Encoder: {encoder_path}")
    print(f"   Decoder: {decoder_path}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Training complete!")
    print("="*70)
    
    return history, full_model, encoder, decoder


def main():
    # Command-line interface
    
    parser = argparse.ArgumentParser(description='Train CNN+LSTM caption model')
    
    parser.add_argument('--name', type=str, default='cnn_lstm_learned',
                        help='Model name for saving')
    parser.add_argument('--embedding', type=str, default='learned',
                        choices=['learned', 'tfidf', 'glove', 'word2vec'],
                        help='Embedding type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension (for learned/tfidf only)')
    parser.add_argument('--lstm_units', type=int, default=256,
                        help='LSTM hidden units')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Image size (square)')
    parser.add_argument('--output_dir', type=str, default='models/approach_1',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_name=args.name,
        embedding_type=args.embedding,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        image_size=(args.image_size, args.image_size),
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

