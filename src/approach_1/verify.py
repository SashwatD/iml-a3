import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from src.utils.dataset_torch import load_vocab
from src.approach_1.basic.caption_model import CNNLSTMModel
from src.approach_1.optimized.caption_model import OptimizedCNNLSTMModel
from src.approach_1.powerful.caption_model import PowerfulCNNLSTMModel

def generate_caption(model, image_path, vocab, variant="basic", max_length=50, device="cpu", image_size=(224, 224), beam_size=5, temperature=0.8, alpha=0.8):
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    # Get True Data
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/sampled_images/artemis_dataset_release_v0.csv")
    painting_name = image_path.split("/")[-1].split(".")[0]
    try:
        df = pd.read_csv(CSV_PATH)
        row = df[df["painting"] == painting_name].iloc[0]
        true_caption = row["utterance"]
    except:
        true_caption = "Unknown"

    with torch.no_grad():
        # 1. Extract Features
        img_features = model.cnn(img)
        
        # Beam Search Setup
        sequences = [[0.0, [vocab.stoi["<start>"]], None]] # Score, Sequence, States
        unk_idx = vocab.stoi.get("<unk>", -1)
        
        if variant == "basic":
            # Basic: Project once
            img_features = model.visual_projection(img_features) # (1, embed_dim)
            sequences = [[0.0, [vocab.stoi["<start>"]], None]]
            
        elif variant in ["optimized", "powerful"]:
            # Optimized/Powerful: Input Feeding + State Init
            
            # 1. Init States
            # h0, c0 = init_h(img), init_c(img)
            h0 = torch.tanh(model.init_h(img_features)).unsqueeze(0).repeat(2, 1, 1)
            c0 = torch.tanh(model.init_c(img_features)).unsqueeze(0).repeat(2, 1, 1)
            initial_states = (h0, c0)
            
            # 2. Prepare Image Context (for Input Feeding)
            img_context = model.visual_proj(img_features).unsqueeze(1) # (1, 1, embed_dim)
            
            sequences = [[0.0, [vocab.stoi["<start>"]], initial_states]]

        for _ in range(max_length):
            all_candidates = []
            
            for score, seq, states in sequences:
                if seq[-1] == vocab.stoi["<end>"]:
                    all_candidates.append([score, seq, states])
                    continue
                
                # Prepare input for this step
                last_token = seq[-1]
                inputs = torch.tensor([last_token]).unsqueeze(0).to(device) # (1, 1)
                embeds = model.embedding(inputs) # (1, 1, embed_dim)
                
                if variant == "basic":
                    # Basic: Concatenate image features at every step
                    img_features_expanded = img_features.unsqueeze(1)
                    lstm_input = torch.cat((embeds, img_features_expanded), dim=2)
                else:
                    # Optimized/Powerful: Input Feeding (Concat Image to Word)
                    # embeds: (1, 1, embed_dim)
                    # img_context: (1, 1, embed_dim)
                    lstm_input = torch.cat((embeds, img_context), dim=2) # (1, 1, 2*embed_dim)
                
                # LSTM Step
                hiddens, new_states = model.lstm(lstm_input, states)
                outputs = model.fc_out(hiddens) # (1, 1, vocab_size)
                
                # Logits for the last step
                logits = outputs[:, -1, :]
                
                # Ban <unk>
                if unk_idx != -1:
                    logits[:, unk_idx] = float('-inf')
                
                # Repetition Penalty
                for prev_token in set(seq):
                    logits[:, prev_token] /= 1.2
                
                # Temperature
                logits = logits / temperature
                log_probs = F.log_softmax(logits, dim=1)
                
                # Top-K
                topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                
                for i in range(beam_size):
                    token = topk_ids[0, i].item()
                    prob = topk_probs[0, i].item()
                    
                    all_candidates.append([score + prob, seq + [token], new_states])
            
            # Sort by Score with Length Penalty
            ordered = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            
            if all(seq[-1] == vocab.stoi["<end>"] for _, seq, _ in sequences):
                break
                
        best_seq = sequences[0][1]
        
        # Decode
        result_caption = []
        for token in best_seq:
            if token == vocab.stoi["<start>"]: continue
            if token == vocab.stoi["<end>"]: break
            result_caption.append(vocab.itos[token])
            
    return " ".join(result_caption), true_caption

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
            model = PowerfulCNNLSTMModel(vocab_size=len(vocab), embed_dim=512, hidden_dim=512)
            
        model = model.to(device)
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("CRITICAL: You must RETRAIN the model because the architecture has changed (Input Feeding).")
            sys.exit(1)
            
        # Select Random Image
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            row = df.sample(1).iloc[0]
            img_rel_path = os.path.join(row['art_style'], row['painting'] + '.jpg')
            full_img_path = os.path.join(BASE_DIR, "data/sampled_images/wikiart", img_rel_path)
            
            if os.path.exists(full_img_path):
                print(f"Selected Image: {full_img_path}")
                
                caption, true_caption = generate_caption(
                    model, 
                    full_img_path, 
                    vocab, 
                    variant=VARIANT, 
                    device=device,
                    beam_size=5,
                    alpha=0.8
                )
                print(f"True Caption: {true_caption}")
                print(f"Generated: {caption}")
            else:
                print(f"Image not found: {full_img_path}")
        else:
            print("CSV not found.")
    else:
        print(f"Model or vocab not found at {MODEL_PATH}")
