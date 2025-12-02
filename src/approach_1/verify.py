import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.utils.dataset_torch import load_vocab
from src.approach_1.basic.caption_model import CNNLSTMModel
from src.approach_1.optimized.caption_model import OptimizedCNNLSTMModel
from src.approach_1.powerful.caption_model import PowerfulCNNLSTMModel

def generate_caption(model, image_path, vocab, variant="basic", max_length=50, device="cpu", image_size=(224, 224)):
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 1. Extract Features
        img_features = model.cnn(img)
        
        if variant == "basic":
            # Basic: Project and Concatenate at every step
            img_features = model.visual_projection(img_features) # (1, embed_dim)
            
            states = None
            inputs = torch.tensor([vocab.stoi["<start>"]]).unsqueeze(0).to(device) # (1, 1)
            
            result_caption = []
            
            for _ in range(max_length):
                embeds = model.embedding(inputs) # (1, 1, embed_dim)
                img_features_expanded = img_features.unsqueeze(1)
                lstm_input = torch.cat((embeds, img_features_expanded), dim=2) # (1, 1, 2*embed_dim)
                
                hiddens, states = model.lstm(lstm_input, states)
                outputs = model.fc_out(hiddens)
                
                predicted = outputs.argmax(2).item()
                if predicted == vocab.stoi["<end>"]: break
                
                result_caption.append(vocab.itos[predicted])
                inputs = torch.tensor([predicted]).unsqueeze(0).to(device)
                
        elif variant in ["optimized", "powerful"]:
            # Optimized/Powerful: Image as Context Primer (First Token)
            img_embed = model.visual_projector(img_features) # (1, embed_dim)
            
            # Step 1: Feed Image
            lstm_input = img_embed.unsqueeze(1) # (1, 1, embed_dim)
            hiddens, states = model.lstm(lstm_input)
            
            # Step 2: Feed <start> to generate first word
            current_token_idx = vocab.stoi["<start>"]
            result_caption = []
            
            for _ in range(max_length):
                token_embed = model.embedding(torch.tensor([[current_token_idx]]).to(device))
                hiddens, states = model.lstm(token_embed, states)
                
                outputs = model.fc_out(hiddens)
                predicted_idx = outputs.argmax(2).item()
                
                if predicted_idx == vocab.stoi["<end>"]: break
                
                result_caption.append(vocab.itos[predicted_idx])
                current_token_idx = predicted_idx
                
    return " ".join(result_caption)

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/sampled_images/artemis_dataset_release_v0.csv")
    
    # Configuration
    VARIANT = "powerful" # "basic", "optimized", "powerful"
    EMBEDDING_TYPE = "tfidf"
    
    if VARIANT == "basic":
        MODEL_DIR = os.path.join(BASE_DIR, f"models/approach-1-basic/{EMBEDDING_TYPE}")
    elif VARIANT == "optimized":
        MODEL_DIR = os.path.join(BASE_DIR, f"models/approach-1-optimized/{EMBEDDING_TYPE}")
    elif VARIANT == "powerful":
        MODEL_DIR = os.path.join(BASE_DIR, f"models/approach-1-powerful/{EMBEDDING_TYPE}")
        
    MODEL_PATH = os.path.join(MODEL_DIR, "model_final.pth")
    VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Variant: {VARIANT}")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        vocab = load_vocab(VOCAB_PATH)
        
        if VARIANT == "basic":
            model = CNNLSTMModel(vocab_size=len(vocab), embed_dim=300, hidden_dim=512)
        elif VARIANT == "optimized":
            model = OptimizedCNNLSTMModel(vocab_size=len(vocab), embed_dim=300, hidden_dim=512)
        elif VARIANT == "powerful":
            model = PowerfulCNNLSTMModel(vocab_size=len(vocab), embed_dim=300, hidden_dim=512)
            
        model = model.to(device)
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
            
        # Select Random Image
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            row = df.sample(1).iloc[0]
            img_rel_path = os.path.join(row['art_style'], row['painting'] + '.jpg')
            full_img_path = os.path.join(BASE_DIR, "data/sampled_images/wikiart", img_rel_path)
            
            if os.path.exists(full_img_path):
                print(f"Selected Image: {full_img_path}")
                print(f"True Caption: {row['utterance']}")
                
                caption = generate_caption(model, full_img_path, vocab, variant=VARIANT, device=device)
                print(f"Generated: {caption}")
            else:
                print(f"Image not found: {full_img_path}")
        else:
            print("CSV not found.")
    else:
        print(f"Model or vocab not found at {MODEL_PATH}")
