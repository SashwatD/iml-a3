import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel



class FlashViTCaptionModel(nn.Module):
    def __init__(
        self, 
        vocab_size=5000, 
        embed_dim=512, 
        num_heads=8, 
        num_decoder_layers=6, 
        ff_dim=2048, 
        dropout=0.4, 
        max_length=50,
        embedding_matrix=None,
        num_emotions=9,
        vit_model_path='/home/arnav.goyal_ug2023/iml/iml-a3/downloads/google_vit_local'
    ):
        super().__init__()
        
        # Auto-adjust num_heads if not divisible
        if embed_dim % num_heads != 0:
            # Find valid divisors
            valid_heads = [h for h in range(1, 17) if embed_dim % h == 0] # Check up to 16 heads
            if not valid_heads:
                valid_heads = [1]
            # Pick closest
            new_heads = min(valid_heads, key=lambda x: abs(x - num_heads))
            print(f"Embed_dim {embed_dim} not divisible by num_heads {num_heads}. Auto-adjusting num_heads to {new_heads}.")
            num_heads = new_heads
        
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
        
        # Native PyTorch Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
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
        
        # Causal Mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(SeqLen).to(captions.device)
        
        # Native Decoder Forward
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
            
        caption_logits = self.fc_out(dec_out)
        return caption_logits, emotion_logits