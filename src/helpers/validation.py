import sys
import os
import argparse
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.utils.dataset_torch import load_vocab
from src.helpers.metrics import evaluate_model

# Import Models
from src.approach_1.basic.caption_model import CNNLSTMModel
from src.approach_1.powerful.caption_model import PowerfulCNNLSTMModel
from src.approach_2.pretrained.caption_model import PretrainedViTCaptionModel
from src.approach_2.flash_attention.caption_model import FlashViTCaptionModel
from src.approach_2.finetuning.caption_model import FinetunedViTCaptionModel

def generate_caption_approach_1(model, img_tensor, vocab, variant, max_length=50, device="cpu", beam_size=5, alpha=0.8):
    model.eval()
    with torch.no_grad():
        # 1. Extract Features
        img_features = model.cnn(img_tensor)
        
        # Beam Search Setup
        sequences = [[0.0, [vocab.stoi["<start>"]], None]] # Score, Sequence, States
        unk_idx = vocab.stoi.get("<unk>", -1)
        
        if variant == "basic":
            img_features = model.visual_projection(img_features)
            sequences = [[0.0, [vocab.stoi["<start>"]], None]]
        elif variant == "powerful":
            # Init States
            h0 = torch.tanh(model.init_h(img_features)).unsqueeze(0).repeat(2, 1, 1)
            c0 = torch.tanh(model.init_c(img_features)).unsqueeze(0).repeat(2, 1, 1)
            initial_states = (h0, c0)
            img_context = model.visual_proj(img_features).unsqueeze(1)
            sequences = [[0.0, [vocab.stoi["<start>"]], initial_states]]

        for _ in range(max_length):
            all_candidates = []
            for score, seq, states in sequences:
                if seq[-1] == vocab.stoi["<end>"]:
                    all_candidates.append([score, seq, states])
                    continue
                
                last_token = seq[-1]
                inputs = torch.tensor([last_token]).unsqueeze(0).to(device)
                embeds = model.embedding(inputs)
                
                if variant == "basic":
                    img_features_expanded = img_features.unsqueeze(1)
                    lstm_input = torch.cat((embeds, img_features_expanded), dim=2)
                else:
                    lstm_input = torch.cat((embeds, img_context), dim=2)
                
                hiddens, new_states = model.lstm(lstm_input, states)
                outputs = model.fc_out(hiddens)
                logits = outputs[:, -1, :]
                
                if unk_idx != -1: logits[:, unk_idx] = float('-inf')
                for prev_token in set(seq): logits[:, prev_token] /= 1.2
                
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                
                for i in range(beam_size):
                    token = topk_ids[0, i].item()
                    prob = topk_probs[0, i].item()
                    all_candidates.append([score + prob, seq + [token], new_states])
            
            ordered = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            if all(seq[-1] == vocab.stoi["<end>"] for _, seq, _ in sequences): break
            
        best_seq = sequences[0][1]
        caption = [vocab.itos[token] for token in best_seq if token not in [vocab.stoi["<start>"], vocab.stoi["<end>"]]]
        
        # Emotion Prediction (Powerful only)
        emotion_logits = None
        if variant == "powerful":
            emotion_logits = model.emotion_head(img_features)
            
        return " ".join(caption), emotion_logits

def generate_caption_approach_2(model, img_tensor, vocab, variant, max_length=50, device="cpu", beam_size=5, alpha=0.8):
    model.eval()
    with torch.no_grad():
        vit_out = model.vit(img_tensor).last_hidden_state
        enc_out = model.visual_projection(vit_out)
            
        # Decoder (Beam Search)
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
                
                dec_out = tgt_emb
                if variant == "flash":
                    dec_out = model.decoder(dec_out, enc_out, tgt_mask=tgt_mask)
                else:
                    for layer in model.decoder_layers: dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
                
                outputs = model.fc_out(dec_out)
                logits = outputs[:, -1, :]
                
                if unk_idx != -1: logits[:, unk_idx] = float('-inf')
                for prev_token in set(seq): logits[:, prev_token] /= 1.2
                
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                
                for i in range(beam_size):
                    token = topk_ids[0, i].item()
                    prob = topk_probs[0, i].item()
                    all_candidates.append([score + prob, seq + [token]])
            
            ordered = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
            sequences = ordered[:beam_size]
            if all(seq[-1] == vocab.stoi["<end>"] for _, seq in sequences): break
            
        best_seq = sequences[0][1]
        caption = [vocab.itos[token] for token in best_seq if token not in [vocab.stoi["<start>"], vocab.stoi["<end>"]]]
        
        # Emotion Prediction (Flash only)
        emotion_logits = None
        if variant == "flash":
             # Recalculate global features for emotion head
             global_features = enc_out.mean(dim=1)
             emotion_logits = model.emotion_head(global_features)

        return " ".join(caption), emotion_logits


def validate_model(
    approach,
    variant,
    model_path,
    vocab_path,
    embedding_type,
    csv_path="data/sampled_images/artemis_dataset_release_v0.csv",
    img_root="data/sampled_images/wikiart",
    num_samples=1000
):
    # Embedding Dimension Map
    EMBED_DIM_MAP = {
        "tfidf": 512,
        "word2vec": 300,
        "word2vec-google-news-300": 300,
        "glove": 100,
        "glove-wiki-gigaword-100": 100,
        "glove-wiki-gigaword-300": 300
    }
    
    # Determine embedding dim
    embed_dim = 300 # Default
    for key, val in EMBED_DIM_MAP.items():
        if key in embedding_type:
            embed_dim = val
            break
            
    print(f"Using embedding dimension: {embed_dim} for type {embedding_type}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Vocab
    print(f"Loading vocab from {vocab_path}...")
    vocab = load_vocab(vocab_path)
    
    # 2. Load Model
    print(f"Loading model from {model_path}...")
    if approach == 1:
        if variant == "basic":
            model = CNNLSTMModel(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=512)
        elif variant == "powerful":
            model = PowerfulCNNLSTMModel(vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=512)
    elif approach == 2:
        if variant == "pretrained":
            model = PretrainedViTCaptionModel(vocab_size=len(vocab))
        elif variant == "scratch":
            model = ViTCaptionModel(image_size=256, vocab_size=len(vocab))
        elif variant == "flash":
            vit_path = os.path.abspath("downloads/google_vit_local") # Ensure absolute path
            model = FlashViTCaptionModel(vocab_size=len(vocab), vit_model_path=vit_path, embed_dim=embed_dim)
        elif variant == "finetuning":
            model = FinetunedViTCaptionModel(vocab_size=len(vocab))
            
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False) 
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Try adjusting model parameters in validation.py to match training config.")
        return None
        
    # 3. Prepare Data
    print("Preparing validation data...")
    df = pd.read_csv(csv_path)
    
    if len(df) < num_samples:
        print(f"Warning: CSV has fewer samples ({len(df)}) than requested ({num_samples}). Using all.")
        samples = df
    else:
        samples = df.sample(num_samples, random_state=42)
        
    print(f"Validating on {len(samples)} samples...")
    
    # Transforms
    if approach == 2 and variant in ["pretrained", "finetuning", "flash"]:
        image_size = (224, 224)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        image_size = (224, 224)
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    EMOTION_MAP = {
        "amusement": 0, "anger": 1, "awe": 2, "contentment": 3,
        "disgust": 4, "excitement": 5, "fear": 6, "sadness": 7,
        "something else": 8
    }
    
    true_captions = []
    generated_captions = []
    true_emotions = []
    pred_emotions = []
    
    for idx, row in tqdm(samples.iterrows(), total=len(samples)):
        img_rel_path = os.path.join(row['art_style'], row['painting'] + '.jpg')
        full_img_path = os.path.join(img_root, img_rel_path)
        
        if not os.path.exists(full_img_path):
            continue
            
        try:
            img = Image.open(full_img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Generate
            if approach == 1:
                gen_cap, emo_logits = generate_caption_approach_1(model, img_tensor, vocab, variant, device=device)
            else:
                gen_cap, emo_logits = generate_caption_approach_2(model, img_tensor, vocab, variant, device=device)
                
            true_captions.append(row['utterance'])
            generated_captions.append(gen_cap)
            
            # Emotions
            true_emo_idx = EMOTION_MAP.get(row['emotion'], 8)
            true_emotions.append(true_emo_idx)
            
            if emo_logits is not None:
                pred_emo_idx = torch.argmax(emo_logits, dim=-1).item()
                pred_emotions.append(pred_emo_idx)
            else:
                pred_emotions.append(8) # Default/Placeholder
                
        except Exception as e:
            print(f"Error processing {full_img_path}: {e}")
            continue
            
    # Calculate Metrics
    print("\nCalculating Metrics...")
    
    # Convert emotions to tensors
    true_emotions = torch.tensor(true_emotions)
    pred_emotions = torch.tensor(pred_emotions)
    
    # If model didn't predict emotions (all 8s or None), we might want to skip emotion metrics
    # But evaluate_model handles it.
    
    metrics = evaluate_model(true_captions, generated_captions, emotion_preds=pred_emotions, emotion_targets=true_emotions)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Image Captioning Model")
    parser.add_argument("--approach", type=int, choices=[1, 2], required=True, help="Approach 1 or 2")
    parser.add_argument("--variant", type=str, required=True, help="Model variant (basic, powerful, pretrained, flash, finetuning)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.pkl")
    parser.add_argument("--csv_path", type=str, default="data/sampled_images/artemis_dataset_release_v0.csv", help="Path to dataset CSV")
    parser.add_argument("--img_root", type=str, default="data/sampled_images/wikiart", help="Root directory of images")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--embedding_type", type=str, required=True, help="Embedding type (tfidf, word2vec, glove)")
    
    args = parser.parse_args()
    
    metrics = validate_model(
        approach=args.approach,
        variant=args.variant,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        embedding_type=args.embedding_type,
        csv_path=args.csv_path,
        img_root=args.img_root,
        num_samples=args.num_samples
    )
    
    if metrics:
        print("\nFinal Metrics:")
        print(metrics)
