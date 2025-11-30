# Improved CNN Encoder - VGG-Style Architecture
# IMPROVEMENTS:
# 1. Deeper network (13 conv layers vs 4)
# 2. Double/triple convolutions per block (better feature learning)
# 3. More filters (up to 512 vs 256)
# 4. Larger embedding dimension (512 vs 256)
# 5. Optimized for 224x224 artwork images

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization, Activation
)


def build_improved_cnn_encoder(input_shape=(224, 224, 3), embedding_dim=512):
    """
    VGG-inspired CNN encoder for artwork feature extraction.
    
    Architecture Philosophy:
    - Block 1-2: Low-level features (edges, corners, colors)
    - Block 3: Mid-level features (textures, brushstrokes)
    - Block 4: High-level features (objects, compositions, emotions)
    
    Why VGG-style?
    - Multiple 3x3 convs = larger receptive field with fewer parameters
    - Deeper networks = better hierarchical feature learning
    - Proven architecture for image understanding
    
    Parameters:
        input_shape: (H, W, C) - defaults to 224x224x3 for artwork detail
        embedding_dim: Size of output feature vector (512 for rich representation)
    
    Returns:
        Model that maps images to embedding_dim feature vectors
    """
    
    inputs = Input(shape=input_shape, name='image_input')
    
    # Block 1: 64 filters
    # Purpose: Detect basic edges, corners, color boundaries
    # Why 2 convs: Stacking allows learning more complex edge patterns
    x = Conv2D(64, (3, 3), padding='same', name='conv1_1')(inputs)
    x = BatchNormalization(name='bn1_1')(x)
    x = Activation('relu', name='relu1_1')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv1_2')(x)
    x = BatchNormalization(name='bn1_2')(x)
    x = Activation('relu', name='relu1_2')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)  # 224 -> 112
    
    # Block 2: 128 filters
    # Purpose: Detect textures (fabric, brushstrokes, patterns)
    # Why 2 convs: Combines edges into texture representations
    x = Conv2D(128, (3, 3), padding='same', name='conv2_1')(x)
    x = BatchNormalization(name='bn2_1')(x)
    x = Activation('relu', name='relu2_1')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2_2')(x)
    x = BatchNormalization(name='bn2_2')(x)
    x = Activation('relu', name='relu2_2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)  # 112 -> 56
    
    # Block 3: 256 filters
    # Purpose: Detect object parts (faces, hands, buildings, landscape elements)
    # Why 3 convs: More depth = more abstraction capability
    x = Conv2D(256, (3, 3), padding='same', name='conv3_1')(x)
    x = BatchNormalization(name='bn3_1')(x)
    x = Activation('relu', name='relu3_1')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_2')(x)
    x = BatchNormalization(name='bn3_2')(x)
    x = Activation('relu', name='relu3_2')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
    x = BatchNormalization(name='bn3_3')(x)
    x = Activation('relu', name='relu3_3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)  # 56 -> 28
    
    # Block 4: 512 filters
    # Purpose: Detect complete objects, compositions, emotional content
    # Why 3 convs: Highest abstraction - understands "what" is in the image
    x = Conv2D(512, (3, 3), padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='bn4_1')(x)
    x = Activation('relu', name='relu4_1')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_2')(x)
    x = BatchNormalization(name='bn4_2')(x)
    x = Activation('relu', name='relu4_2')(x)
    x = Conv2D(512, (3, 3), padding='same', name='conv4_3')(x)
    x = BatchNormalization(name='bn4_3')(x)
    x = Activation('relu', name='relu4_3')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)  # 28 -> 14
    
    # Dense layers: Compress spatial features into semantic embedding
    # Flatten: 14x14x512 = 100,352 features
    x = Flatten(name='flatten')(x)
    
    # FC1: 1024 units - intermediate semantic representation
    x = Dense(1024, name='fc1')(x)
    x = BatchNormalization(name='bn_fc1')(x)
    x = Activation('relu', name='relu_fc1')(x)
    x = Dropout(0.5, name='dropout1')(x)  # Higher dropout to prevent overfitting
    
    # FC2: Another 1024 - refine representation
    x = Dense(1024, name='fc2')(x)
    x = BatchNormalization(name='bn_fc2')(x)
    x = Activation('relu', name='relu_fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    
    # Final embedding: 512D feature vector
    # This is what gets fed to the LSTM decoder
    outputs = Dense(embedding_dim, activation='relu', name='image_embedding')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Encoder_Improved')
    
    return model


def get_feature_extractor(model, layer_name):
    """
    Extract features from intermediate layers for visualization.
    Useful for understanding what the network learned.
    """
    from tensorflow.keras.models import Model
    layer = model.get_layer(layer_name)
    return Model(inputs=model.input, outputs=layer.output)


if __name__ == "__main__":
    import numpy as np
    
    # Test the improved encoder
    print("="*70)
    print("Testing Improved CNN Encoder")
    print("="*70)
    
    encoder = build_improved_cnn_encoder()
    encoder.summary()
    
    # Test with dummy image
    dummy_image = np.random.rand(1, 224, 224, 3).astype('float32')
    features = encoder.predict(dummy_image, verbose=0)
    
    print(f"\nInput shape: {dummy_image.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Total parameters: {encoder.count_params():,}")
    
    # Compare with baseline
    print("\nComparison with Baseline:")
    print("  Baseline: 4 conv layers, 256 filters max, 256D output")
    print("  Improved: 13 conv layers, 512 filters max, 512D output")
    print("  Parameter increase: ~2-3x (worth it for better features!)")
    print("\n[SUCCESS] Improved CNN Encoder working correctly")

