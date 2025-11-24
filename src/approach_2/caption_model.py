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

    def call(self, inputs, encoder_outputs, training=False, use_causal_mask=False):
        # Self-attention
        # If use_causal_mask is True, we let MHA handle it or we construct it.
        
        # To be safe and explicit:
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
    def __init__(self, maxlen, vocab_size, embed_dim, embedding_matrix=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        if embedding_matrix is not None:
            self.token_emb = layers.Embedding(
                input_dim=vocab_size, 
                output_dim=embed_dim,
                weights=[embedding_matrix],
                trainable=False # Freeze pre-trained embeddings
            )
        else:
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
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
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
    dropout_rate=0.1,
    embedding_matrix=None # New argument
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
    x = TokenAndPositionEmbedding(
        max_length, vocab_size, projection_dim, embedding_matrix=embedding_matrix
    )(caption_inputs)
    
    for _ in range(transformer_layers):
        x = TransformerDecoderBlock(
            projection_dim, num_heads, ff_dim, dropout_rate
        )(x, encoded_patches, use_causal_mask=True)

    # Output
    outputs = layers.Dense(vocab_size)(x)
    
    model = models.Model(inputs=[inputs, caption_inputs], outputs=outputs)
    return model

def masked_loss(y_true, y_pred):
    # Calculate loss while ignoring padding tokens (0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    mask = tf.math.not_equal(y_true, 0)
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_acc_percent(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, y_pred.dtype)
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return 100.0 * tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
