import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x) # (B, E, H/P, W/P)
        x = x.flatten(2) # (B, E, N)
        x = x.transpose(1, 2) # (B, N, E)
        x = x + self.position_embedding
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_outputs, tgt_mask=None, tgt_key_padding_mask=None):
        # Self-attention
        attn1, _ = self.att1(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout1(attn1))
        
        # Cross-attention
        attn2, _ = self.att2(x, encoder_outputs, encoder_outputs)
        x = self.norm2(x + self.dropout2(attn2))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class ViTCaptionModel(nn.Module):
    def __init__(
        self, 
        image_size=256, 
        patch_size=16, 
        vocab_size=5000, 
        embed_dim=256, 
        num_heads=4, 
        num_encoder_layers=4, 
        num_decoder_layers=4, 
        ff_dim=512, 
        dropout=0.1,
        max_length=50,
        embedding_matrix=None,
        num_emotions=9
    ):
        super().__init__()
        
        # Encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim=embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Auxiliary Emotion Head
        self.emotion_head = nn.Linear(embed_dim, num_emotions)
        
        # Decoder
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
             self.token_emb.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
             self.token_emb.weight.requires_grad = False # Freeze if pretrained
             
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, images, captions):
        # Encoder
        enc_out = self.patch_embed(images)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
            
        # Auxiliary Emotion Prediction (Global Average Pooling)
        # enc_out is (B, N, E)
        global_features = enc_out.mean(dim=1)
        emotion_logits = self.emotion_head(global_features)
            
        # Decoder
        B, SeqLen = captions.shape
        # Create position indices
        positions = torch.arange(0, SeqLen).unsqueeze(0).repeat(B, 1).to(captions.device)
        
        tgt_emb = self.token_emb(captions) + self.pos_emb(positions)
        tgt_emb = self.dropout(tgt_emb)
        
        # Causal Mask
        tgt_mask = torch.triu(torch.ones(SeqLen, SeqLen) * float('-inf'), diagonal=1).to(captions.device)
        
        # Padding Mask (if 0 is pad)
        # tgt_key_padding_mask = (captions == 0) 
        
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
            
        caption_logits = self.fc_out(dec_out)
        return caption_logits, emotion_logits
