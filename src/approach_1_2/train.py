# Training Script for Improved CNN+LSTM Model
# Includes all optimizations: deeper CNN, larger capacity, better training

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import pickle
import json
from datetime import datetime
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, CSVLogger, LearningRateScheduler
)

from src.approach_1_2.caption_model import (
    build_improved_caption_model,
    compile_improved_model,
    get_lr_schedule
)
from src.approach_1_2.data_loader import ImprovedArtEmisDataGenerator
from src.helpers.tokenizer import CaptionTokenizer


def train_improved_model(
    embedding_type='learned',
    epochs=30,
    batch_size=16,  # Reduced for 224x224 images
    learning_rate=0.001,
    model_name=None,
    use_more_data=False
):
    """
    Train improved image captioning model.
    
    KEY IMPROVEMENTS:
    1. Deeper CNN (VGG-style, 13 layers)
    2. Larger capacity (512 units vs 256)
    3. Higher dropout (0.5 vs 0.3)
    4. Larger images (224x224 vs 128x128)
    5. Gradient clipping (clipnorm=1.0)
    6. Learning rate warmup + decay
    7. Data augmentation
    8. More training data (optional)
    
    Parameters:
        embedding_type: 'learned', 'tfidf', 'word2vec', or 'glove'
        epochs: Number of training epochs (30 default)
        batch_size: Batch size (16 for 224x224 images)
        learning_rate: Initial learning rate (0.001)
        model_name: Custom model name (auto-generated if None)
        use_more_data: Use 30k samples instead of 12k (if available)
    """
    
    print("="*70)
    print("IMPROVED CNN+LSTM TRAINING")
    print("="*70)
    
    # Configuration
    data_dir = Path('data/processed')
    models_dir = Path('models/approach_1_2')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model parameters (IMPROVED)
    vocab_size = 5000
    embedding_dim = 512  # UP from 256
    lstm_units = 512     # UP from 256
    max_length = 50
    dropout_rate = 0.5   # UP from 0.3
    image_size = (224, 224)  # UP from (128, 128)
    
    # Generate model name
    if model_name is None:
        model_name = f"improved_cnn_lstm_{embedding_type}"
    
    print(f"\n[CONFIG] Model: {model_name}")
    print(f"[CONFIG] Embedding: {embedding_type}")
    print(f"[CONFIG] Image size: {image_size}")
    print(f"[CONFIG] Embedding dim: {embedding_dim}")
    print(f"[CONFIG] LSTM units: {lstm_units}")
    print(f"[CONFIG] Dropout: {dropout_rate}")
    print(f"[CONFIG] Batch size: {batch_size}")
    print(f"[CONFIG] Epochs: {epochs}")
    print(f"[CONFIG] Initial LR: {learning_rate}")
    
    # Load tokenizer
    print(f"\n[1/6] Loading tokenizer...")
    tokenizer_path = data_dir / 'tokenizer.pkl'
    tokenizer = CaptionTokenizer.load(tokenizer_path)
    print(f"  Vocabulary size: {len(tokenizer.word_to_index)}")
    
    # Prepare embedding matrix
    print(f"\n[2/6] Preparing embeddings...")
    embedding_matrix = None
    trainable_embeddings = True
    
    if embedding_type == 'learned':
        print("  Using learned embeddings (trained from scratch)")
        embedding_matrix = None
        trainable_embeddings = True
    
    elif embedding_type == 'tfidf':
        print("  Loading TF-IDF embeddings...")
        from src.helpers.embeddings import load_tfidf_embeddings
        embedding_matrix = load_tfidf_embeddings(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            data_dir=data_dir
        )
        trainable_embeddings = False
        print(f"  TF-IDF matrix shape: {embedding_matrix.shape}")
    
    elif embedding_type == 'word2vec':
        print("  Loading Word2Vec embeddings...")
        from src.helpers.embeddings import load_word2vec_embeddings
        embedding_matrix = load_word2vec_embeddings(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim
        )
        trainable_embeddings = False
        print(f"  Word2Vec matrix shape: {embedding_matrix.shape}")
    
    elif embedding_type == 'glove':
        print("  Loading GloVe embeddings...")
        from src.helpers.embeddings import load_glove_embeddings
        embedding_matrix = load_glove_embeddings(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim
        )
        trainable_embeddings = False
        print(f"  GloVe matrix shape: {embedding_matrix.shape}")
    
    # Create data generators
    print(f"\n[3/6] Creating data generators...")
    
    train_generator = ImprovedArtEmisDataGenerator(
        metadata_path=data_dir / 'train_data_metadata.pkl',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        augment=True  # IMPROVEMENT: Data augmentation for training
    )
    
    val_generator = ImprovedArtEmisDataGenerator(
        metadata_path=data_dir / 'val_data_metadata.pkl',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        augment=False  # No augmentation for validation
    )
    
    print(f"  Training samples: {train_generator.n_samples}")
    print(f"  Validation samples: {val_generator.n_samples}")
    print(f"  Training batches: {len(train_generator)}")
    print(f"  Validation batches: {len(val_generator)}")
    
    # Build model
    print(f"\n[4/6] Building improved model...")
    encoder, decoder, full_model = build_improved_caption_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length,
        dropout_rate=dropout_rate,
        embedding_matrix=embedding_matrix,
        trainable_embeddings=trainable_embeddings,
        image_size=image_size
    )
    
    print(f"  Total parameters: {full_model.count_params():,}")
    print(f"  Encoder parameters: {encoder.count_params():,}")
    print(f"  Decoder parameters: {decoder.count_params():,}")
    
    # Compile model with improvements
    print(f"\n[5/6] Compiling model...")
    compile_improved_model(full_model, learning_rate=learning_rate)
    
    # Setup callbacks
    print(f"\n[6/6] Setting up callbacks...")
    
    # Model checkpoint - save best model
    checkpoint_path = models_dir / f'{model_name}_best.h5'
    checkpoint = ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    
    # Early stopping - stop if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,  # More patience for larger model
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # More aggressive (was 0.7)
        patience=3,  # Reduce patience
        min_lr=1e-6,
        verbose=1
    )
    
    # Learning rate schedule with warmup
    lr_schedule = LearningRateScheduler(
        get_lr_schedule(
            initial_lr=learning_rate,
            warmup_epochs=5,
            decay_rate=0.95
        ),
        verbose=1
    )
    
    # CSV logger
    log_path = models_dir / f'{model_name}_training_log.csv'
    csv_logger = CSVLogger(str(log_path))
    
    callbacks = [checkpoint, early_stop, reduce_lr, lr_schedule, csv_logger]
    
    print(f"  Callbacks: ModelCheckpoint, EarlyStopping, ReduceLR, LRScheduler, CSVLogger")
    
    # Save configuration
    config = {
        'model_name': model_name,
        'embedding_type': embedding_type,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'image_size': list(image_size),
        'vocab_size': vocab_size,
        'max_length': max_length,
        'train_samples': train_generator.n_samples,
        'val_samples': val_generator.n_samples,
        'improvements': [
            'VGG-style CNN (13 layers)',
            '512 embedding dim',
            '512 LSTM units',
            '0.5 dropout',
            '224x224 images',
            'Gradient clipping',
            'LR warmup + decay',
            'Data augmentation'
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = models_dir / f'{model_name}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Configuration saved to {config_path}")
    
    # Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"\nExpected improvements over baseline:")
    print(f"  - Baseline accuracy: ~9%")
    print(f"  - Target accuracy: 20-30%")
    print(f"  - Training time: ~4-6 hours (25-30 epochs)")
    print("\n" + "="*70 + "\n")
    
    history = full_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final models
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Saving models...")
    print("="*70)
    
    encoder_path = models_dir / f'{model_name}_encoder.h5'
    decoder_path = models_dir / f'{model_name}_decoder.h5'
    full_path = models_dir / f'{model_name}_final.h5'
    
    encoder.save(str(encoder_path))
    decoder.save(str(decoder_path))
    full_model.save(str(full_path))
    
    print(f"\n[SAVED] Encoder: {encoder_path}")
    print(f"[SAVED] Decoder: {decoder_path}")
    print(f"[SAVED] Full model: {full_path}")
    print(f"[SAVED] Best model: {checkpoint_path}")
    print(f"[SAVED] Training log: {log_path}")
    print(f"[SAVED] Config: {config_path}")
    
    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    print(f"Train Loss: {final_train_loss:.4f}")
    print(f"Val Loss: {final_val_loss:.4f}")
    print(f"Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"Val Accuracy: {final_val_acc*100:.2f}%")
    print(f"Overfitting Gap: {(final_train_loss - final_val_loss):.4f}")
    print("="*70)
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train improved image captioning model')
    parser.add_argument('--embedding', type=str, default='learned',
                       choices=['learned', 'tfidf', 'word2vec', 'glove'],
                       help='Embedding type')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name')
    
    args = parser.parse_args()
    
    try:
        history = train_improved_model(
            embedding_type=args.embedding,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_name=args.name
        )
        print("\n[SUCCESS] Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

