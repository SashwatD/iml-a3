# LSTM Decoder for Caption Generation
# Generates captions word-by-word using image features

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    RepeatVector, Concatenate, TimeDistributed
)


def build_caption_decoder(vocab_size, 
                         embedding_dim=256, 
                         lstm_units=256,
                         max_length=50,
                         dropout_rate=0.3,
                         embedding_matrix=None,
                         trainable_embeddings=True):
    
    # Input 1: Image features from CNN (256D)
    image_input = Input(shape=(embedding_dim,), name='image_features')
    
    # Input 2: Partial caption sequence
    caption_input = Input(shape=(max_length,), name='caption_input')
    
    # Repeat image features for each timestep
    image_features = RepeatVector(max_length)(image_input)
    
    # Word embeddings: convert word indices to dense vectors
    # Can be initialized with pre-trained weights or trained from scratch
    if embedding_matrix is not None:
        # Use pre-trained embeddings (TF-IDF, GloVe, Word2Vec)
        word_embeddings = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],  # Initialize with pre-trained
            trainable=trainable_embeddings,  # Usually False for pre-trained
            mask_zero=True,
            name='word_embedding'
        )(caption_input)
    else:
        # Learn embeddings from scratch (random initialization)
        word_embeddings = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='word_embedding'
        )(caption_input)
    
    # Combine image features with word embeddings
    combined = Concatenate(axis=-1, name='combine')([image_features, word_embeddings])
    
    # LSTM Layer 1: learn sequential patterns
    lstm1_output = LSTM(lstm_units, return_sequences=True, name='lstm1')(combined)
    lstm1_output = Dropout(dropout_rate, name='dropout1')(lstm1_output)
    
    # LSTM Layer 2: refine understanding
    lstm2_output = LSTM(lstm_units, return_sequences=True, name='lstm2')(lstm1_output)
    lstm2_output = Dropout(dropout_rate, name='dropout2')(lstm2_output)
    
    # Output layer: predict next word at each timestep
    output = TimeDistributed(
        Dense(vocab_size, name='output_dense'),
        name='word_prediction'
    )(lstm2_output)
    
    model = Model(
        inputs=[image_input, caption_input],
        outputs=output,
        name='Caption_Decoder'
    )
    
    return model


def generate_caption(encoder, decoder, image, tokenizer, max_length=50):
    # Generate caption using greedy search
    import numpy as np
    
    # Extract image features
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    image_features = encoder.predict(image, verbose=0)
    
    # Start with <start> token
    start_token = tokenizer.word_to_index.get('<start>', 1)
    end_token = tokenizer.word_to_index.get('<end>', 2)
    caption_indices = [start_token]
    
    # Generate words one by one
    for _ in range(max_length):
        sequence = np.zeros((1, max_length))
        sequence[0, :len(caption_indices)] = caption_indices
        
        # Predict next word
        predictions = decoder.predict([image_features, sequence], verbose=0)
        next_word_pred = predictions[0, len(caption_indices)-1, :]
        next_word_idx = np.argmax(next_word_pred)
        
        # Stop if <end> token
        if next_word_idx == end_token:
            break
        
        caption_indices.append(next_word_idx)
    
    # Convert indices to words
    caption_words = []
    for idx in caption_indices[1:]:  # Skip <start>
        word = tokenizer.index_to_word.get(idx, '<unk>')
        if word not in ['<start>', '<end>', '<pad>']:
            caption_words.append(word)
    
    return ' '.join(caption_words)


if __name__ == "__main__":
    import numpy as np
    
    # Test the decoder
    vocab_size = 5000
    decoder = build_caption_decoder(vocab_size=vocab_size)
    decoder.summary()
    
    # Test with dummy data
    dummy_image_features = np.random.rand(2, 256).astype('float32')
    dummy_caption = np.random.randint(0, vocab_size, size=(2, 50))
    
    predictions = decoder.predict([dummy_image_features, dummy_caption], verbose=0)
    
    print(f"\nImage features: {dummy_image_features.shape}")
    print(f"Caption input: {dummy_caption.shape}")
    print(f"Predictions: {predictions.shape}")
    print(f"Total parameters: {decoder.count_params():,}")
    print("[SUCCESS] LSTM Decoder working correctly")
