import tensorflow as tf
from tensorflow.keras import layers, models

class PatchCreation(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate
        })
        return config

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training=False, mask=None):
        # Self-attention (causal mask is usually handled by the caller or implicitly if using Sequential, 
        # but here we might need to pass it. For simplicity in this block, we assume inputs are already masked 
        # or we rely on the standard Keras masking if passed).
        # However, for caption generation, we need causal masking.
        
        # Note: Keras MultiHeadAttention supports 'use_causal_mask' in newer versions, 
        # or we can pass a boolean mask.
        
        attn1 = self.att1(inputs, inputs, attention_mask=mask) 
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        
        # Cross-attention
        attn2 = self.att2(out1, encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att1.key_dim,
            "num_heads": self.att1.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.pos_emb.input_dim,
            "vocab_size": self.token_emb.input_dim,
            "embed_dim": self.token_emb.output_dim
        })
        return config

def build_vit_caption_model(
    input_shape=(256, 256, 3),
    patch_size=16,
    num_patches=256, # (256//16)**2
    projection_dim=256,
    num_heads=4,
    transformer_layers=4,
    vocab_size=5000,
    max_length=50,
    ff_dim=512,
    dropout_rate=0.1
):
    # --- Encoder (ViT) ---
    inputs = layers.Input(shape=input_shape)
    patches = PatchCreation(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        encoded_patches = TransformerEncoderBlock(
            projection_dim, num_heads, ff_dim, dropout_rate
        )(encoded_patches)
    
    # --- Decoder ---
    caption_inputs = layers.Input(shape=(max_length,), dtype="int64")
    x = TokenAndPositionEmbedding(max_length, vocab_size, projection_dim)(caption_inputs)
    
    # Create causal mask
    # (batch_size, max_length, max_length)
    # We can use a custom layer or lambda for this if needed, but MHA handles it if we pass use_causal_mask=True
    # or construct it manually. Let's construct manually for clarity/compatibility.
    
    # Actually, Keras MHA `attention_mask` argument:
    # "Boolean mask of shape (B, T, S)..."
    # For causal, we want to mask future tokens.
    
    # Let's use a simpler approach: The DecoderBlock will take the mask.
    # But constructing the mask inside the model definition is cleaner.
    
    def causal_attention_mask(batch_size, n_dest, n_src, dtype):
        """
        Mask the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    # We will let the MHA layer handle causal masking if we set use_causal_mask=True (TF 2.10+)
    # Or we can just pass the mask.
    # Given the environment might vary, let's rely on the custom block logic.
    # In this implementation, I'll assume the user might want to run this on standard Colab/Local.
    
    # To keep it simple and robust:
    # We will just pass the encoder outputs to the decoder.
    # The decoder block needs to handle the causal masking internally or we pass it.
    
    # Let's iterate the decoder layers
    for _ in range(transformer_layers):
        x = TransformerDecoderBlock(
            projection_dim, num_heads, ff_dim, dropout_rate
        )(x, encoded_patches, use_causal_mask=True) # Assuming we modify Block to accept this or handle it

    # Output
    outputs = layers.Dense(vocab_size)(x)
    
    model = models.Model(inputs=[inputs, caption_inputs], outputs=outputs)
    return model

# Re-defining Decoder Block to support use_causal_mask argument properly if passed to call
# Or we can just use the built-in use_causal_mask of MHA if available.
# Let's update the TransformerDecoderBlock to be more robust.

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training=False, use_causal_mask=False):
        # Self-attention
        attn1 = self.att1(inputs, inputs, use_causal_mask=use_causal_mask) 
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        
        # Cross-attention
        attn2 = self.att2(out1, encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)
