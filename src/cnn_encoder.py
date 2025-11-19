# CNN Encoder for Image Feature Extraction
# Extracts 256D feature vectors from 224x224 artwork images

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization, Activation
)


def build_cnn_encoder(input_shape=(224, 224, 3), embedding_dim=256):
    # Input layer
    inputs = Input(shape=input_shape, name='image_input')
    
    # Block 1: 32 filters - detect edges and simple patterns
    x = Conv2D(32, (3, 3), padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)  # 224x224 -> 112x112
    
    # Block 2: 64 filters - detect textures
    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='relu2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)  # 112x112 -> 56x56
    
    # Block 3: 128 filters - detect object parts
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu', name='relu3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)  # 56x56 -> 28x28
    
    # Block 4: 256 filters - detect complete objects
    x = Conv2D(256, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu', name='relu4')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)  # 28x28 -> 14x14
    
    # Flatten and dense layers
    x = Flatten(name='flatten')(x)  # 14x14x256 = 50,176
    x = Dense(512, name='fc1')(x)
    x = BatchNormalization(name='bn_fc')(x)
    x = Activation('relu', name='relu_fc')(x)
    x = Dropout(0.3, name='dropout')(x)  # Prevent overfitting
    
    # Final embedding: 256D feature vector
    outputs = Dense(embedding_dim, activation='relu', name='image_embedding')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Encoder')
    return model


def get_feature_extractor(model, layer_name):
    # Extract features from intermediate layers for visualization
    from tensorflow.keras.models import Model
    layer = model.get_layer(layer_name)
    return Model(inputs=model.input, outputs=layer.output)


if __name__ == "__main__":
    import numpy as np
    
    # Test the encoder
    encoder = build_cnn_encoder()
    encoder.summary()
    
    # Test with dummy image
    dummy_image = np.random.rand(1, 224, 224, 3).astype('float32')
    features = encoder.predict(dummy_image, verbose=0)
    
    print(f"\nInput shape: {dummy_image.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Total parameters: {encoder.count_params():,}")
    print("✓ CNN Encoder working correctly")
