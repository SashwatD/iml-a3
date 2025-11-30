import torch
import torch.nn as nn
from transformers import ViTModel

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

    def forward(self, x, encoder_outputs, tgt_mask=None):
        # Self-attention
        attn1, _ = self.att1(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn1))
        
        # Cross-attention
        attn2, _ = self.att2(x, encoder_outputs, encoder_outputs)
        x = self.norm2(x + self.dropout2(attn2))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class PretrainedViTCaptionModel(nn.Module):
    def __init__(
        self, 
        vocab_size=5000, 
        embed_dim=512, # Decoder dimension
        num_heads=8, 
        num_decoder_layers=4, 
        ff_dim=2048, 
        dropout=0.1,
        max_length=50,
        embedding_matrix=None
    ):
        super().__init__()
        
        # Encoder: Pretrained ViT
        # google/vit-base-patch16-224 outputs 768 dim
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.vit.parameters():
            param.requires_grad = False # Freeze
            
        self.visual_projection = nn.Linear(768, embed_dim)
        
        # Decoder
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
             self.token_emb.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
             self.token_emb.weight.requires_grad = False
             
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, images, captions):
        # images: (B, 3, 224, 224) - PyTorch standard
        # captions: (B, SeqLen)
        
        # Encoder
        vit_out = self.vit(images).last_hidden_state # (B, 197, 768)
        enc_out = self.visual_projection(vit_out) # (B, 197, 512)
        
        # Decoder
        B, SeqLen = captions.shape
        positions = torch.arange(0, SeqLen).unsqueeze(0).repeat(B, 1).to(captions.device)
        
        tgt_emb = self.token_emb(captions) + self.pos_emb(positions)
        tgt_emb = self.dropout(tgt_emb)
        
        tgt_mask = torch.triu(torch.ones(SeqLen, SeqLen) * float('-inf'), diagonal=1).to(captions.device)
        
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
            
        output = self.fc_out(dec_out)
        return output
