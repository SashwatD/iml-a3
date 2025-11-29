# Complete Image Caption Model
# Combines CNN Encoder + LSTM Decoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from src.approach_1.cnn.cnn_encoder import build_cnn_encoder
from src.approach_1.lstm.lstm_decoder import build_caption_decoder


def build_caption_model(vocab_size, 
                       embedding_dim=256,
                       lstm_units=256,
                       max_length=50,
                       dropout_rate=0.3,
                       embedding_matrix=None,
                       trainable_embeddings=True,
                       image_size=(128, 128)):
    
    print("Building Caption Model...")
    
    # Build CNN Encoder
    encoder = build_cnn_encoder(
        input_shape=(*image_size, 3),
        embedding_dim=embedding_dim
    )
    print(f"[CNN Encoder] {encoder.count_params():,} parameters")
    
    # Build LSTM Decoder
    decoder = build_caption_decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length,
        dropout_rate=dropout_rate,
        embedding_matrix=embedding_matrix,
        trainable_embeddings=trainable_embeddings
    )
    print(f"[LSTM Decoder] {decoder.count_params():,} parameters")
    
    # Connect encoder and decoder
    image_input = Input(shape=(*image_size, 3), name='image')
    caption_input = Input(shape=(max_length,), name='caption')
    
    image_features = encoder(image_input)
    output = decoder([image_features, caption_input])
    
    full_model = Model(
        inputs=[image_input, caption_input],
        outputs=output,
        name='Caption_Model'
    )
    
    print(f"[Total] {full_model.count_params():,} parameters")
    
    return encoder, decoder, full_model


def compile_model(model, learning_rate=0.001):
    # Compile model for training with masked loss (ignores padding)
    optimizer = Adam(learning_rate=learning_rate)
    
    # Custom masked loss function
    def masked_loss(y_true, y_pred):
        # Mask padding tokens (index 0)
        mask = tf.math.not_equal(y_true, 0)
        
        loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = loss_fn(y_true, y_pred)
        
        # Apply mask and compute mean
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=['accuracy']
    )
    print("[Model] Compiled with masked loss")


def mask_loss(y_true, y_pred, padding_value=0):
    # Calculate loss while ignoring padding tokens
    mask = tf.math.not_equal(y_true, padding_value)
    mask = tf.cast(mask, dtype=tf.float32)
    
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)


def generate_caption_greedy(encoder, decoder, image, tokenizer, max_length=50):
    # Generate caption using greedy search
    import numpy as np
    
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    image_features = encoder.predict(image, verbose=0)
    
    start_idx = tokenizer.word_to_index.get('<start>', 1)
    end_idx = tokenizer.word_to_index.get('<end>', 2)
    
    caption_indices = [start_idx]
    
    for _ in range(max_length):
        sequence = np.zeros((1, max_length))
        sequence[0, :len(caption_indices)] = caption_indices
        
        predictions = decoder.predict([image_features, sequence], verbose=0)
        next_word_pred = predictions[0, len(caption_indices)-1, :]
        next_word_idx = np.argmax(next_word_pred)
        
        if next_word_idx == end_idx:
            break
        
        caption_indices.append(next_word_idx)
    
    caption = tokenizer.sequence_to_caption(caption_indices)
    return caption


if __name__ == "__main__":
    import numpy as np
    
    # Test complete model with different embedding types
    print("="*70)
    print("Testing Caption Model")
    print("="*70)
    
    vocab_size = 5000
    image_size = (128, 128)
    
    # Test 1: Learned embeddings
    print("\n1. Testing with learned embeddings...")
    encoder, decoder, full_model = build_caption_model(
        vocab_size=vocab_size,
        image_size=image_size
    )
    compile_model(full_model)
    
    # Test forward pass
    dummy_images = np.random.rand(2, *image_size, 3).astype('float32')
    dummy_captions = np.random.randint(0, vocab_size, size=(2, 50))
    
    predictions = full_model.predict({'image': dummy_images, 'caption': dummy_captions}, verbose=0)
    
    print(f"   Images: {dummy_images.shape}")
    print(f"   Captions: {dummy_captions.shape}")
    print(f"   Predictions: {predictions.shape}")
    
    # Test 2: Pre-trained embeddings
    print("\n2. Testing with pre-trained embeddings...")
    embedding_dim = 100
    fake_embedding_matrix = np.random.rand(vocab_size, embedding_dim).astype('float32')
    
    encoder2, decoder2, full_model2 = build_caption_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        image_size=image_size,
        embedding_matrix=fake_embedding_matrix,
        trainable_embeddings=False
    )
    
    print(f"   Embedding layer trainable: {decoder2.get_layer('word_embedding').trainable}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Complete model working correctly")
    print("="*70)
