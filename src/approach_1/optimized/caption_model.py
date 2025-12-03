import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, output_dim=512):
        super(CustomCNN, self).__init__()
        
        # 5 VGG-Style Blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # (B, 512)
        return x

class OptimizedCNNLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, embedding_matrix=None, num_emotions=9, dropout_rate=0.4):
        super(OptimizedCNNLSTMModel, self).__init__()
        
        # 1. Shared Encoder
        self.cnn = CustomCNN(output_dim=512)
        
        # 2. Emotion Branch (Auxiliary)
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )
        
        # 3. Caption Branch
        
        # A. Visual Features Projection
        # We project image features to match embedding dimension
        self.visual_proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # B. Initialization Projectors
        # We transform the image to initialize LSTM hidden/cell states
        self.init_h = nn.Linear(512, hidden_dim)
        self.init_c = nn.Linear(512, hidden_dim)
        
        # C. Text Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            # FIX: Unfreeze embeddings so the model can learn specific art words
            self.embedding.weight.requires_grad = True 
            
        # D. LSTM (Input Feeding)
        # Input Size = Word_Embed (300) + Image_Proj (300) = 600
        self.lstm = nn.LSTM(
            input_size=embed_dim + embed_dim, 
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images, captions):
        # 1. Get Visual Features
        cnn_feats = self.cnn(images) # (B, 512)
        
        # 2. Predict Emotion (Auxiliary Task)
        emotion_logits = self.emotion_head(cnn_feats)
        
        # 3. Prepare LSTM Inputs
        
        # A. Initialize States with Image
        # This gives the LSTM a "mood" before it sees any text
        h0 = torch.tanh(self.init_h(cnn_feats)).unsqueeze(0).repeat(1, 1, 1) # Repeat for 1 layer
        c0 = torch.tanh(self.init_c(cnn_feats)).unsqueeze(0).repeat(1, 1, 1)
        
        # B. Prepare Image Context (for Input Feeding)
        # (B, 512) -> (B, 300) -> (B, 1, 300)
        img_context = self.visual_proj(cnn_feats).unsqueeze(1)
        
        # C. Prepare Text
        # (B, SeqLen) -> (B, SeqLen, 300)
        # Note: We assume captions input excludes <end>, or we handle it in training loop
        embeds = self.embedding(captions) 
        
        # D. Input Feeding: Concatenate Image to Every Word
        # We expand image context to match sequence length
        seq_len = embeds.size(1)
        img_context_expanded = img_context.repeat(1, seq_len, 1) # (B, SeqLen, 300)
        
        # New Input: (B, SeqLen, 600)
        lstm_input = torch.cat((embeds, img_context_expanded), dim=2)
        
        # 4. LSTM Pass
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        
        # 5. Output
        caption_logits = self.fc_out(self.dropout(lstm_out))
        
        return caption_logits, emotion_logits

    def generate(self, image, vocab, max_len=20):
        # Inference Logic for Input Feeding
        self.eval()
        with torch.no_grad():
            # 1. Encode Image
            cnn_feats = self.cnn(image.unsqueeze(0))
            
            # 2. Init States
            num_layers = self.lstm.num_layers
            h = torch.tanh(self.init_h(cnn_feats)).unsqueeze(0).repeat(num_layers, 1, 1)
            c = torch.tanh(self.init_c(cnn_feats)).unsqueeze(0).repeat(num_layers, 1, 1)
            
            # 3. Prepare Image Context
            img_context = self.visual_proj(cnn_feats).unsqueeze(1) # (1, 1, 300)
            
            # 4. Start Token
            word_idx = vocab.stoi["<start>"]
            caption = []
            
            for _ in range(max_len):
                # Embed Word
                word_embed = self.embedding(torch.tensor([[word_idx]]).to(image.device)) # (1, 1, 300)
                
                # Concat [Word, Image]
                lstm_in = torch.cat((word_embed, img_context), dim=2)
                
                # Step
                out, (h, c) = self.lstm(lstm_in, (h, c))
                logits = self.fc_out(out.squeeze(1))
                
                # Pick Next Word
                word_idx = logits.argmax(1).item()
                if word_idx == vocab.stoi["<end>"]:
                    break
                    
                caption.append(vocab.itos[word_idx])
                
            return " ".join(caption)
