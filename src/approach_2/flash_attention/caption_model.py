import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class FlashAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, is_causal=False):
        # x: (B, L, E)
        # context: (B, S, E) - if None, self-attention
        
        B, L, E = x.shape
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        if context is None:
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            S = context.shape[1]
            k = self.k_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            
        # Flash Attention
        # scaled_dot_product_attention handles is_causal automatically if supported
        # It expects (B, H, L, D)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=self.dropout.p if self.training else 0.0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        x = residual + attn_output
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class FlashDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = FlashAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
        self.cross_attn = FlashAttentionBlock(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, x, encoder_outputs):
        # Self Attention (Causal)
        x = self.self_attn(x, is_causal=True)
        # Cross Attention
        x = self.cross_attn(x, context=encoder_outputs, is_causal=False)
        return x

class FlashViTCaptionModel(nn.Module):
    def __init__(
        self, 
        vocab_size=5000, 
        embed_dim=512, # Increased default
        num_heads=8, # Increased default
        num_decoder_layers=6, # Increased default
        ff_dim=2048, # Increased default
        dropout=0.4, # Increased to 0.4 for stronger regularization
        max_length=50,
        embedding_matrix=None,
        num_emotions=9,
        vit_model_path='/home/arnav.goyal_ug2023/iml/iml-a3/downloads/google_vit_local'
    ):
        super().__init__()
        
        # Encoder: Pretrained ViT
        print(f"Loading ViT from: {vit_model_path}")
        self.vit = ViTModel.from_pretrained(vit_model_path)


        for param in self.vit.parameters():
            param.requires_grad = False # Freeze
            
        self.visual_projection = nn.Linear(768, embed_dim)
        
        # Auxiliary Emotion Head
        self.emotion_head = nn.Linear(embed_dim, num_emotions)
        
        # Decoder
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
             self.token_emb.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
             self.token_emb.weight.requires_grad = False
             
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.decoder_layers = nn.ModuleList([
            FlashDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, images, captions):
        # Encoder
        vit_out = self.vit(images).last_hidden_state # (B, 197, 768)
        enc_out = self.visual_projection(vit_out) # (B, 197, E)
            
        # Auxiliary Emotion Prediction (Global Average Pooling)
        # enc_out is (B, L, E)
        global_features = enc_out.mean(dim=1)
        emotion_logits = self.emotion_head(global_features)
            
        # Decoder
        B, SeqLen = captions.shape
        positions = torch.arange(0, SeqLen).unsqueeze(0).repeat(B, 1).to(captions.device)
        
        tgt_emb = self.token_emb(captions) + self.pos_emb(positions)
        tgt_emb = self.dropout(tgt_emb)
        
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)
            
        caption_logits = self.fc_out(dec_out)
        return caption_logits, emotion_logits