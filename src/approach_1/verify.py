import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.utils.dataset_torch import load_vocab
from src.approach_1.basic.caption-model import CNNLSTMModel

def generate_caption(model, image_path, vocab, max_length=50, device="cpu", image_size=(224, 224)):
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
        img_features = model.visual_projection(img_features) # (1, embed_dim)
        
        # 2. Generate
        states = None
        inputs = torch.tensor([vocab.stoi["<start>"]]).unsqueeze(0).to(device) # (1, 1)
        
        result_caption = []
        
        for _ in range(max_length):
            embeds = model.embedding(inputs) # (1, 1, embed_dim)
            
            # Concatenate image features to input
            # img_features is (1, embed_dim), need (1, 1, embed_dim)
            img_features_expanded = img_features.unsqueeze(1)
            
            lstm_input = torch.cat((embeds, img_features_expanded), dim=2) # (1, 1, 2*embed_dim)
            
            hiddens, states = model.lstm(lstm_input, states)
            outputs = model.fc_out(hiddens) # (1, 1, vocab_size)
            
            predicted = outputs.argmax(2).item()
            
            if predicted == vocab.stoi["<end>"]:
                break
                
            result_caption.append(vocab.itos[predicted])
            inputs = torch.tensor([predicted]).unsqueeze(0).to(device)
            
    return " ".join(result_caption)

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/sampled_images/artemis_dataset_release_v0.csv")
    
    # Configuration
    EMBEDDING_TYPE = "tfidf"
    MODEL_PATH = os.path.join(BASE_DIR, f"models/approach-1-basic/{EMBEDDING_TYPE}/model_final.pth")
    VOCAB_PATH = os.path.join(BASE_DIR, f"models/approach-1-basic/{EMBEDDING_TYPE}/vocab.pkl")
    IMAGE_SIZE = (224, 224)
    EMBED_DIM = 300
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        vocab = load_vocab(VOCAB_PATH)
        
        model = CNNLSTMModel(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            hidden_dim=512
        ).to(device)
        
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
                
                caption = generate_caption(model, full_img_path, vocab, device=device, image_size=IMAGE_SIZE)
                print(f"Generated: {caption}")
            else:
                print(f"Image not found: {full_img_path}")
        else:
            print("CSV not found.")
    else:
        print(f"Model or vocab not found at {MODEL_PATH}")
