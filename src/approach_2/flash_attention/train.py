import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.preprocessing.preprocessing import load_and_clean_data, create_tf_dataset, save_vectorizer
from src.approach_2.flash_attention.caption_model import build_vit_caption_model, masked_loss, masked_acc_percent
from src.approach_2.flash_attention.embeddings import get_tfidf_embeddings, get_pretrained_embeddings

def train_model(
    csv_path,
    image_dir,
    output_dir="models/approach-2/flash_attention",
    sample_size=None,
    batch_size=128,
    epochs=40,
    vocab_size=10000,
    max_length=50,
    image_size=(256, 256),
    use_mixed_precision=True,
    embedding_type="learned", # learned, tfidf, glove, word2vec
    embedding_dim=512
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Enable Mixed Precision
    if use_mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled.")
    
    # 1. Load and Preprocess Data
    df = load_and_clean_data(csv_path, image_dir, sample_size=sample_size)
    
    train_ds, val_ds, vectorizer = create_tf_dataset(
        df, 
        image_size=image_size, 
        batch_size=batch_size, 
        vocab_size=vocab_size, 
        max_length=max_length,
        augment=True # Enable augmentation
    )
    
    # Save vectorizer
    save_vectorizer(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
    
    # 2. Prepare Embeddings
    embedding_matrix = None
    if embedding_type == "learned":
        print("Using learned embeddings (default).")
    elif embedding_type == "tfidf":
        # Note: TF-IDF requires raw text which we don't easily have in the tf.dataset pipeline here.
        # For now, we'll use the dataframe's captions.
        captions = df['utterance'].tolist()
        embedding_matrix = get_tfidf_embeddings(vectorizer, captions, embedding_dim=embedding_dim, vocab_size=vocab_size) 
    elif embedding_type == "glove":
        embedding_matrix = get_pretrained_embeddings(
            vectorizer, 
            model_name="glove-wiki-gigaword-100", # 100d is standard small glove
            embedding_dim=embedding_dim
        )
    elif embedding_type == "word2vec":
        embedding_matrix = get_pretrained_embeddings(
            vectorizer, 
            model_name="word2vec-google-news-300", 
            embedding_dim=embedding_dim
        )
    
    # 3. Build Model
    print(f"Building Vision Transformer Model with {embedding_type} embeddings...")
    model = build_vit_caption_model(
        input_shape=image_size + (3,),
        vocab_size=vocab_size,
        max_length=max_length - 1, # Input sequence is shifted
        transformer_layers=6,
        num_heads=8,
        projection_dim=embedding_dim, # Use embedding_dim
        ff_dim=2048,
        dropout_rate=0.2,
        embedding_matrix=embedding_matrix
    )
    
    # Crucial for Transformers: Linear Warmup -> Cosine Decay
    
    total_steps = num_train_steps * epochs
    warmup_steps = int(0.1 * total_steps) # 10% warmup
    
    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,    # Peak LR
        decay_steps=total_steps,
        alpha=0.01,                    # Minimum LR (1% of peak)
        warmup_target=3e-4,
        warmup_steps=warmup_steps
    )

    # 3. Compile Model
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=0.05,
        beta_1=0.9,
        beta_2=0.95,  # Slightly lower beta_2 for stability
        epsilon=1e-6
    )

    if use_mixed_precision:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_acc_percent],
        jit_compile=True
    )
    
    model.summary()
    
    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1
        )
    ]
    
    # 5. Train
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Save final model
    model.save_weights(os.path.join(output_dir, "final_model.weights.h5"))
    
    return history, model, vectorizer

def plot_history(history, output_dir):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['masked_acc_percent']
    val_acc = history.history['val_masked_acc_percent']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    print(f"Training history plot saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    CSV_PATH = "data/images/artemis_dataset_release_v0.csv"
    IMG_DIR = "data/images/wikiart"
    
    if os.path.exists(CSV_PATH) and os.path.exists(IMG_DIR):
        history, model, vectorizer = train_model(
            CSV_PATH, 
            IMG_DIR, 
            sample_size=5000, # Start small for testing
            epochs=30,
            use_mixed_precision=False
        )
        plot_history(history, "models/approach-2")
    else:
        print("Dataset not found. Please check paths.")
