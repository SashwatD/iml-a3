import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import logging
from datetime import datetime

from src.utils.dataset_torch import load_vocab
from src.approach_1.powerful.caption_model import PowerfulCNNLSTMModel
from src.approach_2.flash_attention.caption_model import FlashViTCaptionModel
from transformers import ViTModel
from unittest.mock import patch

# Setup Logging
log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def load_model(approach, embedding_type, model_path, vocab_path, device):
    vocab = load_vocab(vocab_path)
    
    if approach == 1:
        model = PowerfulCNNLSTMModel(
            vocab_size=len(vocab),
            embed_dim=512 if embedding_type == "tfidf" else 300 if "word2vec" in embedding_type else 100,
            hidden_dim=512
        ).to(device)
    elif approach == 2:
        original_loader = ViTModel.from_pretrained
        def patched_loader(*args, **kwargs):
            kwargs["attn_implementation"] = "eager"
            return original_loader(*args, **kwargs)
            
        with patch("transformers.ViTModel.from_pretrained", side_effect=patched_loader):
            model = FlashViTCaptionModel(
                vocab_size=len(vocab),
                embed_dim=512 if embedding_type == "tfidf" else 300 if "word2vec" in embedding_type else 100,
                num_heads=4 if embedding_type == "glove-wiki-gigaword-100" else 8,
                ff_dim=2048,
                num_decoder_layers=6,
                vit_model_path="downloads/google_vit_local"
            ).to(device)
            
    # Load weights
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
                model.load_state_dict(state.get("model_state_dict", state.get("state_dict")))
            else:
                model.load_state_dict(state)
            model.eval()
            return model, vocab
        except Exception as e:
            logging.error(f"Error loading weights for {model_path}: {e}")
            return None, None
    else:
        logging.error(f"Model path not found: {model_path}")
        return None, None

def generate_caption(model, image, vocab, device, max_length=50, beam_size=5, temperature=0.8, alpha=0.8):
    model.eval()
    
    # Preprocessing
    # Approach 1 uses [0.5, 0.5, 0.5] mean/std normalization
    # Approach 2 uses [0.5, 0.5, 0.5] as well (checked train.py)
    # But verify.py for A2 uses [0.5, 0.5, 0.5]
    # Let's stick to standard ImageNet mean/std if unsure, but verify.py uses 0.5.
    # Let's use 0.5 as per verify.py files.
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # --- Approach 1: CNN-LSTM ---
        if isinstance(model, PowerfulCNNLSTMModel):
            # 1. Extract Features
            img_features = model.cnn(img_tensor)
            
            # 2. Init States
            num_layers = 2
            h0 = torch.tanh(model.init_h(img_features)).unsqueeze(0).repeat(num_layers, 1, 1)
            c0 = torch.tanh(model.init_c(img_features)).unsqueeze(0).repeat(num_layers, 1, 1)
            initial_states = (h0, c0)
            
            # 3. Prepare Image Context
            img_context = model.visual_proj(img_features).unsqueeze(1) # (1, 1, embed_dim)
            
            # Beam Search Setup
            sequences = [[0.0, [vocab.stoi["<start>"]], initial_states]]
            unk_idx = vocab.stoi.get("<unk>", -1)
            
            for _ in range(max_length):
                all_candidates = []
                for score, seq, states in sequences:
                    if seq[-1] == vocab.stoi["<end>"]:
                        all_candidates.append([score, seq, states])
                        continue
                    
                    last_token = seq[-1]
                    inputs = torch.tensor([last_token]).unsqueeze(0).to(device)
                    embeds = model.embedding(inputs)
                    
                    # Input Feeding
                    lstm_input = torch.cat((embeds, img_context), dim=2)
                    
                    hiddens, new_states = model.lstm(lstm_input, states)
                    outputs = model.fc_out(hiddens)
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
                    
                    topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                    
                    for i in range(beam_size):
                        token = topk_ids[0, i].item()
                        prob = topk_probs[0, i].item()
                        all_candidates.append([score + prob, seq + [token], new_states])
                
                ordered = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
                sequences = ordered[:beam_size]
                
                if all(seq[-1] == vocab.stoi["<end>"] for _, seq, _ in sequences):
                    break
            
            best_seq = sequences[0][1]
            
            # Emotion Prediction
            emotion_logits = model.emotion_head(img_features)
            emotion_idx = emotion_logits.argmax(1).item()

        # --- Approach 2: Vision Transformer ---
        elif isinstance(model, FlashViTCaptionModel):
            # 1. Encoder Pass
            vit_out = model.vit(img_tensor).last_hidden_state
            enc_out = model.visual_projection(vit_out)
            
            # Beam Search Setup
            sequences = [[0.0, [vocab.stoi["<start>"]]]]
            unk_idx = vocab.stoi.get("<unk>", -1)
            
            for _ in range(max_length):
                all_candidates = []
                for score, seq in sequences:
                    if seq[-1] == vocab.stoi["<end>"]:
                        all_candidates.append([score, seq])
                        continue
                    
                    inputs = torch.tensor([seq]).to(device)
                    B, SeqLen = inputs.shape
                    positions = torch.arange(0, SeqLen).unsqueeze(0).to(device)
                    
                    tgt_emb = model.token_emb(inputs) + model.pos_emb(positions)
                    tgt_mask = torch.triu(torch.ones(SeqLen, SeqLen) * float('-inf'), diagonal=1).to(device)
                    
                    dec_out = model.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask)
                    outputs = model.fc_out(dec_out)
                    logits = outputs[:, -1, :]
                    
                    if unk_idx != -1:
                        logits[:, unk_idx] = float('-inf')
                    
                    for prev_token in set(seq):
                        logits[:, prev_token] /= 1.2
                    
                    logits = logits / temperature
                    log_probs = F.log_softmax(logits, dim=1)
                    
                    topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                    
                    for i in range(beam_size):
                        token = topk_ids[0, i].item()
                        prob = topk_probs[0, i].item()
                        all_candidates.append([score + prob, seq + [token]])
                
                ordered = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
                sequences = ordered[:beam_size]
                
                if all(seq[-1] == vocab.stoi["<end>"] for _, seq in sequences):
                    break
            
            best_seq = sequences[0][1]
            
            # Emotion Prediction
            # Re-run forward pass with best sequence for emotion head
            # Note: FlashViTCaptionModel forward returns (caption_logits, emotion_logits)
            # But we need global features for emotion head.
            # In forward(): global_features = enc_out.mean(dim=1); emotion_logits = self.emotion_head(global_features)
            # We already have enc_out.
            global_features = enc_out.mean(dim=1)
            emotion_logits = model.emotion_head(global_features)
            emotion_idx = emotion_logits.argmax(1).item()

        # Decode Caption
        caption = [vocab.itos[idx] for idx in best_seq if vocab.itos[idx] not in ["<start>", "<end>", "<pad>"]]
        return " ".join(caption), emotion_idx

def evaluate_single_image(image_path, models_config, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return

    logging.info(f"\nEvaluating Image: {image_path}")
    logging.info("-" * 50)

    print(f"\n{'Model':<15} | {'Generated Caption'}")
    print("-" * 80)

    for config in models_config:
        model_name = config['name']
        model = config['model']
        vocab = config['vocab']
        
        if model is None:
            continue
            
        generated_cap, pred_emotion_idx = generate_caption(model, image, vocab, device)
        
        # Map emotion index back to string
        emotion_map = {
            0: 'amusement', 1: 'awe', 2: 'contentment', 3: 'excitement',
            4: 'anger', 5: 'disgust', 6: 'fear', 7: 'sadness', 8: 'something else'
        }
        pred_emotion = emotion_map.get(pred_emotion_idx, "unknown")
        
        # Log to file
        logging.info(f"Model: {model_name}")
        logging.info(f"Caption: {generated_cap}")
        logging.info("-" * 30)
        
        # Print to terminal
        print(f"{model_name:<15} | {generated_cap}")

def main():
    parser = argparse.ArgumentParser(description="Generate Captions and Emotions for a Single Image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the .jpg image file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Define Model Configurations
    configs = [
        {"name": "A1-TFIDF", "approach": 1, "emb": "tfidf", "path": "models/approach-1-powerful/tfidf/model_final.pth", "vocab": "models/approach-1-powerful/tfidf/vocab.pkl"},
        {"name": "A1-Word2Vec", "approach": 1, "emb": "word2vec-google-news-300", "path": "models/approach-1-powerful/word2vec-google-news-300/model_final.pth", "vocab": "models/approach-1-powerful/word2vec-google-news-300/vocab.pkl"},
        {"name": "A1-GloVe", "approach": 1, "emb": "glove-wiki-gigaword-100", "path": "models/approach-1-powerful/glove-wiki-gigaword-100/model_final.pth", "vocab": "models/approach-1-powerful/glove-wiki-gigaword-100/vocab.pkl"},
        {"name": "A2-TFIDF", "approach": 2, "emb": "tfidf", "path": "models/approach-2-flash/tfidf/model_final.pth", "vocab": "models/approach-2-flash/tfidf/vocab.pkl"},
        {"name": "A2-Word2Vec", "approach": 2, "emb": "word2vec-google-news-300", "path": "models/approach-2-flash/word2vec-google-news-300/model_final.pth", "vocab": "models/approach-2-flash/word2vec-google-news-300/vocab.pkl"},
        {"name": "A2-GloVe", "approach": 2, "emb": "glove-wiki-gigaword-100", "path": "models/approach-2-flash/glove-wiki-gigaword-100/model_final.pth", "vocab": "models/approach-2-flash/glove-wiki-gigaword-100/vocab.pkl"},
    ]

    # Load Models
    loaded_models = []
    print("Loading models... This might take a minute.")
    for cfg in configs:
        logging.info(f"Loading {cfg['name']}...")
        model, vocab = load_model(cfg['approach'], cfg['emb'], cfg['path'], cfg['vocab'], device)
        if model:
            loaded_models.append({
                'name': cfg['name'],
                'model': model,
                'vocab': vocab
            })
    
    if not loaded_models:
        print("No models could be loaded. Check paths.")
        return

    evaluate_single_image(args.image_path, loaded_models, device)

if __name__ == "__main__":
    main()
