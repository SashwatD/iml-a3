import sys
import os
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from src.utils.dataset_torch import load_vocab
from src.approach_2.scratch.caption_model import ViTCaptionModel
from src.approach_2.pretrained.caption_model import PretrainedViTCaptionModel
from src.approach_2.flash_attention.caption_model import FlashViTCaptionModel
from src.approach_2.finetuning.caption_model import FinetunedViTCaptionModel

def generate_caption(model, image_path, vocab, variant, max_length=50, device="cpu", image_size=(224, 224), beam_size=5, temperature=0.8):
    model.eval()
    
    # 1. Preprocessing
    if variant in ["pretrained", "finetuning"]:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    
    # Get True Data
    painting_name = image_path.split("/")[-1].split(".")[0]
    try:
        df = pd.read_csv(CSV_PATH)
        row = df[df["painting"] == painting_name].iloc[0]
        true_caption = row["utterance"]
        true_emotion = row["emotion"]
    except:
        true_caption = "Unknown"
        true_emotion = "Unknown"

    with torch.no_grad():
        # Encoder Pass
        if variant in ["pretrained", "finetuning", "flash"]:
            vit_out = model.vit(img).last_hidden_state
            enc_out = model.visual_projection(vit_out)
        elif variant == "scratch":
            enc_out = model.patch_embed(img)
            for layer in model.encoder_layers:
                enc_out = layer(enc_out)
        
        # Decoder Pass (Beam Search with Constraints)
        sequences = [[0.0, [vocab.stoi["<start>"]]]]
        
        # Identify the <unk> token ID to ban it
        unk_idx = vocab.stoi.get("<unk>", -1)
        
        for _ in range(max_length):
            all_candidates = []
            
            for score, seq in sequences:
                # If sequence already ended, keep it
                if seq[-1] == vocab.stoi["<end>"]:
                    all_candidates.append([score, seq])
                    continue
                
                inputs = torch.tensor([seq]).to(device)
                B, SeqLen = inputs.shape
                
                # Create Inputs
                positions = torch.arange(0, SeqLen).unsqueeze(0).to(device)
                
                # Embedding
                tgt_emb = model.token_emb(inputs) + model.pos_emb(positions)

                # Mask
                tgt_mask = torch.triu(torch.ones(SeqLen, SeqLen) * float('-inf'), diagonal=1).to(device)
                
                # Forward Pass
                dec_out = tgt_emb
                for layer in model.decoder_layers:
                    if variant in ["pretrained", "scratch", "finetuning"]:
                        dec_out = layer(dec_out, enc_out, tgt_mask=tgt_mask)
                    elif variant == "flash":
                        dec_out = layer(dec_out, enc_out) 

                outputs = model.fc_out(dec_out) # Shape: (1, SeqLen, Vocab)
                
                # Get logits for the LAST token
                logits = outputs[:, -1, :] 
                
                # Ban <unk>
                if unk_idx != -1:
                    logits[:, unk_idx] = float('-inf')

                # Repetition Penalty
                # Lower the score of tokens already in the sequence
                for prev_token in set(seq):
                    logits[:, prev_token] /= 1.2  # Divide logits (if positive) or multiply (if negative) to discourage
                
                # Apply Temperature
                logits = logits / temperature
                log_probs = F.log_softmax(logits, dim=1)
                
                # Select Top-K
                topk_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                
                for i in range(beam_size):
                    token = topk_ids[0, i].item()
                    prob = topk_probs[0, i].item()
                    
                    # Add to candidates
                    all_candidates.append([score + prob, seq + [token]])
            
            # Sort by Score (Highest is best because log_probs are negative closest to 0)
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = ordered[:beam_size]
            
            # Stop if all beams are finished
            if all(seq[-1] == vocab.stoi["<end>"] for _, seq in sequences):
                break
        
        best_seq = sequences[0][1]
        
        # Decode to words
        result_caption = []
        for token in best_seq:
            if token == vocab.stoi["<start>"]: continue
            if token == vocab.stoi["<end>"]: break
            result_caption.append(vocab.itos[token])
            
    return " ".join(result_caption), true_caption, true_emotion

if __name__ == "__main__":
    VARIANT = "pretrained" # Options: "scratch", "pretrained", "flash", "finetuning"
    RANDOM_SELECTION=True
    
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    CSV_PATH = os.path.join(BASE_DIR, "data/sampled_images/artemis_dataset_release_v0.csv")
    
    if VARIANT == "pretrained":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-pretrained/word2vec/model_epoch_5.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-pretrained/word2vec/vocab.pkl")
        IMAGE_SIZE = (224, 224)
    elif VARIANT == "scratch":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-scratch/tfidf/model_epoch_5.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-scratch/tfidf/vocab.pkl")
        IMAGE_SIZE = (256, 256)
    elif VARIANT == "flash":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-flash/tfidf/model_final.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-flash/tfidf/vocab.pkl")
        IMAGE_SIZE = (256, 256)
    elif VARIANT == "finetuning":
        MODEL_PATH = os.path.join(BASE_DIR, "models/approach-2-finetuning/word2vec/model_final.pth")
        VOCAB_PATH = os.path.join(BASE_DIR, "models/approach-2-finetuning/word2vec/vocab.pkl")
        IMAGE_SIZE = (224, 224)
   
    # Randomly select test image
    if RANDOM_SELECTION:
        IMG_PATH = None
        if os.path.exists(CSV_PATH):
            try:
                df = pd.read_csv(CSV_PATH)
                # Try finding a valid image up to 10 times
                for _ in range(10):
                    row = df.sample(1).iloc[0]
                    img_rel_path = os.path.join(row['art_style'], row['painting'] + '.jpg')
                    full_img_path = os.path.join(BASE_DIR, "data/sampled_images/wikiart", img_rel_path)
                    if os.path.exists(full_img_path):
                        IMG_PATH = full_img_path
                        print(f"Selected Random Image: {IMG_PATH}")
                        break
                if IMG_PATH is None:
                    print("Could not find a valid image after 10 tries.")
                    sys.exit(1)
            except Exception as e:
                print(f"Error reading CSV or selecting image: {e}")
                sys.exit(1)
        else:
            print(f"CSV not found at: {CSV_PATH}")
            sys.exit(1)
    else:
        IMG_PATH = os.path.join(BASE_DIR, "data/image.png")
        if os.path.exists(IMG_PATH):
            print(f"Using Image: {IMG_PATH}")
        else:
            print(f"Image not found at: {IMG_PATH}")
            sys.exit(1)
    
    # Generation Params
    BEAM_SIZE = 5
    TEMPERATURE = 0.8

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Variant: {VARIANT}")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        vocab = load_vocab(VOCAB_PATH)
        
        # Initialize Model
        if VARIANT == "pretrained":
            model = PretrainedViTCaptionModel(vocab_size=len(vocab)).to(device)
        elif VARIANT == "scratch":
            model = ViTCaptionModel(image_size=IMAGE_SIZE[0], vocab_size=len(vocab)).to(device)
        elif VARIANT == "flash":
            model = FlashViTCaptionModel(image_size=IMAGE_SIZE[0], vocab_size=len(vocab)).to(device)
        elif VARIANT == "finetuning":
            model = FinetunedViTCaptionModel(vocab_size=len(vocab)).to(device)
            
        # Load Weights
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Note: If model architecture changed (e.g. added emotion embeddings), you need to retrain.")
            sys.exit(1)
        
        if os.path.exists(IMG_PATH):
            caption, true_caption, emotion = generate_caption(
                model, 
                IMG_PATH, 
                vocab, 
                variant=VARIANT,
                device=device, 
                image_size=IMAGE_SIZE,
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE
            )
            print(f"Generated: {caption}")
            print(f"True Caption: {true_caption}")
            print(f"Emotion: {emotion}")
        else:
            print(f"Image not found: {IMG_PATH}")
    else:
        print(f"Model or vocab not found at:\n{MODEL_PATH}\n{VOCAB_PATH}")
