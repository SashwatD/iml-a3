# Complete Image Caption Model
# Combines CNN Encoder + LSTM Decoder

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from src.cnn_encoder import build_cnn_encoder
from src.lstm_decoder import build_caption_decoder


def build_caption_model(vocab_size, 
                       embedding_dim=256,
                       lstm_units=256,
                       max_length=50,
                       dropout_rate=0.3):
    
    print("Building Caption Model...")
    
    # Build CNN Encoder
    encoder = build_cnn_encoder(
        input_shape=(224, 224, 3),
        embedding_dim=embedding_dim
    )
    print(f"✓ CNN Encoder: {encoder.count_params():,} parameters")
    
    # Build LSTM Decoder
    decoder = build_caption_decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length,
        dropout_rate=dropout_rate
    )
    print(f"✓ LSTM Decoder: {decoder.count_params():,} parameters")
    
    # Connect encoder and decoder
    image_input = Input(shape=(224, 224, 3), name='image')
    caption_input = Input(shape=(max_length,), name='caption')
    
    image_features = encoder(image_input)
    output = decoder([image_features, caption_input])
    
    full_model = Model(
        inputs=[image_input, caption_input],
        outputs=output,
        name='Caption_Model'
    )
    
    print(f"✓ Total parameters: {full_model.count_params():,}")
    
    return encoder, decoder, full_model


def compile_model(model, learning_rate=0.001):
    # Compile model for training
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    print("✓ Model compiled")


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
    
    # Test complete model
    vocab_size = 5000
    encoder, decoder, full_model = build_caption_model(vocab_size=vocab_size)
    compile_model(full_model)
    
    # Test forward pass
    dummy_images = np.random.rand(2, 224, 224, 3).astype('float32')
    dummy_captions = np.random.randint(0, vocab_size, size=(2, 50))
    
    predictions = full_model.predict([dummy_images, dummy_captions], verbose=0)
    
    print(f"\nTest forward pass:")
    print(f"Images: {dummy_images.shape}")
    print(f"Captions: {dummy_captions.shape}")
    print(f"Predictions: {predictions.shape}")
    print("Complete model working correctly")
