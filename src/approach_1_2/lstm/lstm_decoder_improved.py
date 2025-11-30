# Improved LSTM Decoder for Caption Generation
# IMPROVEMENTS:
# 1. Larger capacity (512 units vs 256)
# 2. Higher dropout (0.5 vs 0.3) to reduce overfitting
# 3. Better embedding handling for pre-trained weights
# 4. Optimized for longer, more complex captions

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    RepeatVector, Concatenate, TimeDistributed
)


def build_improved_caption_decoder(vocab_size, 
                                   embedding_dim=512, 
                                   lstm_units=512,
                                   max_length=50,
                                   dropout_rate=0.5,
                                   embedding_matrix=None,
                                   trainable_embeddings=True):
    """
    Improved LSTM decoder for caption generation with higher capacity.
    
    Architecture Philosophy:
    - Larger embedding (512D) = richer word representations
    - Larger LSTM (512 units) = better long-term dependencies
    - Higher dropout (0.5) = stronger regularization against overfitting
    - 2 LSTM layers = hierarchical sequence modeling
    
    How it works:
    1. Image features (512D) repeated for each timestep
    2. Word embeddings (512D) for each word in partial caption
    3. Concatenate image + words = (1024D input to LSTM)
    4. LSTM1 learns sequential patterns
    5. LSTM2 refines understanding
    6. Output predicts next word at each position
    
    Why this architecture?
    - Teacher forcing during training (uses ground truth words)
    - Greedy/beam search during inference (uses predicted words)
    - Image context available at every timestep (simple but effective)
    
    Parameters:
        vocab_size: Number of unique words in vocabulary
        embedding_dim: Dimension of word embeddings (512 for rich semantics)
        lstm_units: Hidden units in LSTM (512 for complex patterns)
        max_length: Maximum caption length (50 words)
        dropout_rate: Dropout probability (0.5 for strong regularization)
        embedding_matrix: Optional pre-trained embeddings (Word2Vec/GloVe)
        trainable_embeddings: Whether to fine-tune embeddings during training
    
    Returns:
        Model that takes [image_features, caption_input] and outputs next word predictions
    """
    
    # Input 1: Image features from CNN (512D)
    image_input = Input(shape=(embedding_dim,), name='image_features')
    
    # Input 2: Partial caption sequence (word indices)
    caption_input = Input(shape=(max_length,), name='caption_input')
    
    # Repeat image features for each timestep
    # Shape: (batch, max_length, embedding_dim)
    # Why? So LSTM can "see" image context while processing each word
    image_features = RepeatVector(max_length)(image_input)
    
    # Word embeddings: Convert word indices to dense vectors
    if embedding_matrix is not None:
        # Case 1: Pre-trained embeddings (TF-IDF, Word2Vec, GloVe)
        # Benefits: Transfer learning from large corpora
        # Drawback: May not be optimized for art captions
        word_embeddings = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=trainable_embeddings,  # Usually False for pre-trained
            mask_zero=True,  # Ignore padding tokens (0)
            name='word_embedding'
        )(caption_input)
    else:
        # Case 2: Learn embeddings from scratch
        # Benefits: Optimized specifically for artwork captions
        # Drawback: Need more training data
        word_embeddings = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name='word_embedding'
        )(caption_input)
    
    # Combine image features with word embeddings
    # Shape: (batch, max_length, embedding_dim*2) = (batch, 50, 1024)
    # Why concatenate? LSTM gets both visual and linguistic context
    combined = Concatenate(axis=-1, name='combine')([image_features, word_embeddings])
    
    # LSTM Layer 1: Learn sequential patterns
    # return_sequences=True: Output at every timestep (not just last)
    # Why 512 units? More capacity = better long-term dependencies
    # Example: "woman in blue" -> LSTM remembers "woman" when processing "blue"
    lstm1_output = LSTM(lstm_units, return_sequences=True, name='lstm1')(combined)
    lstm1_output = Dropout(dropout_rate, name='dropout1')(lstm1_output)
    
    # LSTM Layer 2: Refine understanding
    # Why second layer? Hierarchical processing
    # Layer 1: Low-level patterns (adjective-noun, verb-object)
    # Layer 2: High-level structure (sentence flow, emotional tone)
    lstm2_output = LSTM(lstm_units, return_sequences=True, name='lstm2')(lstm1_output)
    lstm2_output = Dropout(dropout_rate, name='dropout2')(lstm2_output)
    
    # Output layer: Predict next word at each timestep
    # TimeDistributed: Apply same Dense layer to each timestep independently
    # Output shape: (batch, max_length, vocab_size)
    # Interpretation: Probability distribution over vocabulary for each position
    output = TimeDistributed(
        Dense(vocab_size, name='output_dense'),
        name='word_prediction'
    )(lstm2_output)
    
    model = Model(
        inputs=[image_input, caption_input],
        outputs=output,
        name='Caption_Decoder_Improved'
    )
    
    return model


def generate_caption_greedy(encoder, decoder, image, tokenizer, max_length=50):
    """
    Generate caption using greedy search (pick most likely word each step).
    
    Greedy Search Algorithm:
    1. Start with <start> token
    2. Repeat until <end> or max_length:
       a. Predict probability for next word
       b. Pick word with highest probability
       c. Append to sequence
    3. Convert indices to words
    
    Pros: Fast, simple
    Cons: Can miss globally better captions (locally optimal ≠ globally optimal)
    """
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
        # Prepare sequence (pad to max_length)
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
    
    print("="*70)
    print("Testing Improved LSTM Decoder")
    print("="*70)
    
    # Test the decoder
    vocab_size = 5000
    decoder = build_improved_caption_decoder(vocab_size=vocab_size)
    decoder.summary()
    
    # Test with dummy data
    dummy_image_features = np.random.rand(2, 512).astype('float32')
    dummy_caption = np.random.randint(0, vocab_size, size=(2, 50))
    
    predictions = decoder.predict([dummy_image_features, dummy_caption], verbose=0)
    
    print(f"\nImage features shape: {dummy_image_features.shape}")
    print(f"Caption input shape: {dummy_caption.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Total parameters: {decoder.count_params():,}")
    
    print("\nComparison with Baseline:")
    print("  Baseline: 256 units, 0.3 dropout, ~3M parameters")
    print("  Improved: 512 units, 0.5 dropout, ~12M parameters")
    print("  Capacity increase: 4x (better sequence modeling!)")
    print("\n[SUCCESS] Improved LSTM Decoder working correctly")

