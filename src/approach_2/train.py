import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from src.preprocessing.preprocessing import load_and_clean_data, create_tf_dataset, save_vectorizer
from src.approach_2.caption_model import build_vit_caption_model, masked_loss, masked_acc_percent

def train_model(
    csv_path,
    image_dir,
    output_dir="models/approach-2",
    sample_size=None,
    batch_size=32,
    epochs=20,
    vocab_size=5000,
    max_length=50,
    image_size=(256, 256),
    use_mixed_precision=True
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
    
    # 2. Build Model
    print("Building Vision Transformer Model...")
    model = build_vit_caption_model(
        input_shape=image_size + (3,),
        vocab_size=vocab_size,
        max_length=max_length - 1, # Input sequence is shifted
        transformer_layers=4,
        num_heads=4,
        projection_dim=256,
        ff_dim=512,
        dropout_rate=0.1
    )
    
    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    if use_mixed_precision:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_acc_percent]
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
