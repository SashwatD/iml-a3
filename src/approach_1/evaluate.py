# Evaluation Script for CNN+LSTM Models
# Generates captions and calculates BLEU scores

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from tqdm import tqdm
import argparse
from tensorflow.keras.models import load_model

from src.approach_1.data_loader import load_generators
from src.approach_1.caption_model import generate_caption_greedy
from src.helpers.metrics import compute_all_bleu_scores


def generate_captions_batch(encoder, decoder, test_gen, tokenizer, num_samples=None, max_length=50):
    # Generate captions for test set
    # Returns: generated_captions, reference_captions, image_paths
    
    generated = []
    references = []
    image_paths = []
    
    # Determine how many samples to evaluate
    if num_samples is None:
        num_samples = len(test_gen.image_paths)
    num_samples = min(num_samples, len(test_gen.image_paths))
    
    print(f"\nGenerating captions for {num_samples} test images...")
    
    for i in tqdm(range(num_samples), desc="Generating"):
        # Get image and reference caption
        img_path = test_gen.image_paths[i]
        ref_caption = test_gen.raw_captions[i]
        
        # Load and preprocess image
        try:
            img = test_gen._load_image(img_path)
            
            # Generate caption
            generated_caption = generate_caption_greedy(
                encoder, decoder, img, tokenizer, max_length=max_length
            )
            
            generated.append(generated_caption)
            references.append(ref_caption)
            image_paths.append(img_path)
            
        except Exception as e:
            print(f"\nWarning: Failed to process {img_path}: {e}")
            continue
    
    return generated, references, image_paths


def evaluate_model(
    model_dir='models/approach_1',
    model_name='cnn_lstm_learned',
    num_samples=None,
    save_predictions=True
):
    # Main evaluation function
    
    print("="*70)
    print(f"EVALUATING: {model_name}")
    print("="*70)
    
    model_dir = Path(model_dir)
    
    # Step 1: Load configuration
    print("\n1. Loading configuration...")
    config_path = model_dir / f'{model_name}_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"   Model: {config['model_name']}")
    print(f"   Embedding: {config['embedding_type']}")
    print(f"   Vocab size: {config['vocab_size']:,}")
    print(f"   Max length: {config['max_length']}")
    
    # Step 2: Load data generators
    print("\n2. Loading test data...")
    image_size = tuple(config['image_size'])
    train_gen, val_gen, test_gen, tokenizer = load_generators(
        batch_size=1,  # Evaluate one at a time
        image_size=image_size
    )
    print(f"   Test samples: {len(test_gen.image_paths):,}")
    
    # Step 3: Load trained model
    print("\n3. Loading trained model...")
    
    # Try loading best model first, fallback to final
    model_paths = [
        model_dir / f'{model_name}_best.h5',
        model_dir / f'{model_name}_final.h5'
    ]
    
    full_model_path = None
    for path in model_paths:
        if path.exists():
            full_model_path = path
            break
    
    if full_model_path is None:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Rebuild model architecture (avoids custom loss loading issues)
    print("   Rebuilding model architecture...")
    from src.approach_1.caption_model import build_caption_model
    
    encoder, decoder, full_model = build_caption_model(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        lstm_units=config['lstm_units'],
        max_length=config['max_length'],
        dropout_rate=config['dropout_rate'],
        image_size=image_size,
        embedding_matrix=None,  # Weights will be loaded
        trainable_embeddings=True
    )
    
    # Load weights from saved model
    print(f"   Loading weights from: {full_model_path.name}")
    full_model.load_weights(full_model_path)
    
    print(f"   Model loaded successfully")
    
    # Step 4: Generate captions
    print("\n4. Generating captions...")
    generated, references, image_paths = generate_captions_batch(
        encoder, decoder, test_gen, tokenizer,
        num_samples=num_samples,
        max_length=config['max_length']
    )
    
    print(f"   Generated {len(generated)} captions")
    
    # Step 5: Calculate BLEU scores
    print("\n5. Calculating BLEU scores...")
    bleu_scores = compute_all_bleu_scores(references, generated)
    
    print(f"\n   Results:")
    for metric, score in bleu_scores.items():
        print(f"   {metric}: {score:.4f}")
    
    # Step 6: Save predictions
    if save_predictions:
        print("\n6. Saving predictions...")
        predictions_dir = model_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        # Save all predictions to JSON
        predictions = []
        for img_path, ref, gen in zip(image_paths, references, generated):
            predictions.append({
                'image_path': str(img_path),
                'reference': ref,
                'generated': gen
            })
        
        pred_file = predictions_dir / f'{model_name}_predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"   Saved to: {pred_file}")
        
        # Save BLEU scores
        scores_file = predictions_dir / f'{model_name}_scores.json'
        with open(scores_file, 'w') as f:
            json.dump(bleu_scores, f, indent=2)
        print(f"   Scores saved to: {scores_file}")
        
        # Save sample predictions (first 20)
        print(f"\n   Sample predictions (first 10):")
        for i in range(min(10, len(generated))):
            print(f"\n   Image: {image_paths[i]}")
            print(f"   Reference: {references[i][:80]}...")
            print(f"   Generated: {generated[i][:80]}...")
    
    print("\n" + "="*70)
    print("[SUCCESS] Evaluation complete!")
    print("="*70)
    
    return bleu_scores, predictions


def compare_models(model_dir='models/approach_1'):
    # Compare BLEU scores across different models
    
    model_dir = Path(model_dir)
    predictions_dir = model_dir / 'predictions'
    
    if not predictions_dir.exists():
        print("No predictions found. Run evaluation first.")
        return
    
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Find all score files
    score_files = list(predictions_dir.glob('*_scores.json'))
    
    if len(score_files) == 0:
        print("No score files found.")
        return
    
    # Load and display all scores
    results = {}
    for score_file in score_files:
        model_name = score_file.stem.replace('_scores', '')
        with open(score_file, 'r') as f:
            scores = json.load(f)
        results[model_name] = scores
    
    # Print comparison table
    print(f"\n{'Model':<30} {'BLEU-1':<10} {'BLEU-2':<10} {'BLEU-3':<10} {'BLEU-4':<10}")
    print("-" * 70)
    
    for model_name, scores in sorted(results.items()):
        print(f"{model_name:<30} "
              f"{scores['BLEU-1']:<10.4f} "
              f"{scores['BLEU-2']:<10.4f} "
              f"{scores['BLEU-3']:<10.4f} "
              f"{scores['BLEU-4']:<10.4f}")
    
    print("="*70)


def main():
    # Command-line interface
    
    parser = argparse.ArgumentParser(description='Evaluate CNN+LSTM caption model')
    
    parser.add_argument('--model_dir', type=str, default='models/approach_1',
                        help='Model directory')
    parser.add_argument('--model_name', type=str, default='cnn_lstm_learned',
                        help='Model name to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all models in directory')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.model_dir)
    else:
        evaluate_model(
            model_dir=args.model_dir,
            model_name=args.model_name,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()

