# Unified Preprocessing for Both Approaches
# Completes tokenization of existing metadata files

import pickle
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.helpers.tokenizer import CaptionTokenizer


def complete_preprocessing(data_dir='data/processed', vocab_size=5000, max_length=50):
    # Load existing metadata that has raw captions and image paths
    # Add tokenized captions so both approaches can use the data
    
    data_dir = Path(data_dir)
    
    print("="*70)
    print("Unified Preprocessing: Adding Tokenization to Existing Metadata")
    print("="*70)
    
    # Step 1: Load all raw caption data to build vocabulary
    print("\n1. Loading existing metadata files...")
    all_captions = []
    metadata_files = {
        'train': data_dir / 'train_data_metadata.pkl',
        'val': data_dir / 'val_data_metadata.pkl',
        'test': data_dir / 'test_data_metadata.pkl'
    }
    
    # Check if files exist
    for split, filepath in metadata_files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Missing {split} metadata: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   {split}: {len(data['raw_captions'])} samples")
        all_captions.extend(data['raw_captions'])
    
    print(f"   Total captions for vocabulary: {len(all_captions)}")
    
    # Step 2: Build tokenizer vocabulary from ALL captions (train+val+test)
    # This ensures all words in val/test are in vocabulary
    print("\n2. Building vocabulary...")
    tokenizer = CaptionTokenizer(vocab_size=vocab_size, max_length=max_length)
    tokenizer.fit_on_captions(all_captions)
    
    print(f"   Vocabulary size: {len(tokenizer.word_to_index)}")
    print(f"   Max sequence length: {max_length}")
    print(f"   Special tokens: {tokenizer.pad_token}, {tokenizer.start_token}, "
          f"{tokenizer.end_token}, {tokenizer.unk_token}")
    
    # Show top 20 most common words
    if tokenizer.word_counts:
        top_words = tokenizer.word_counts.most_common(20)
        print(f"\n   Top 20 words:")
        for word, count in top_words:
            print(f"      '{word}': {count}")
    
    # Step 3: Save tokenizer
    tokenizer_path = data_dir / 'tokenizer.pkl'
    tokenizer.save(tokenizer_path)
    print(f"\n3. Tokenizer saved to {tokenizer_path}")
    
    # Step 4: Process each split and add tokenized captions
    print("\n4. Tokenizing captions for each split...")
    
    for split, filepath in metadata_files.items():
        # Load existing metadata
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        raw_captions = data['raw_captions']
        image_paths = data['image_paths']
        
        # Tokenize captions (collect all first, then pad in batch)
        captions_input = []  # With <start> token
        captions_target = []  # With <end> token
        
        for caption in raw_captions:
            # Convert caption to sequence of indices (without special tokens)
            sequence = tokenizer.caption_to_sequence(caption, add_special_tokens=False)
            
            # Input: <start> + sequence
            input_seq = [tokenizer.word_to_index['<start>']] + sequence
            
            # Target: sequence + <end>
            target_seq = sequence + [tokenizer.word_to_index['<end>']]
            
            captions_input.append(input_seq)
            captions_target.append(target_seq)
        
        # Pad all sequences at once (more efficient)
        captions_input = tokenizer.pad_sequences(captions_input, padding='post')
        captions_target = tokenizer.pad_sequences(captions_target, padding='post')
        
        # Update metadata with tokenized captions
        data['captions_input'] = captions_input
        data['captions_target'] = captions_target
        
        # Keep image_paths as numpy array for consistency
        if isinstance(image_paths, list):
            data['image_paths'] = np.array(image_paths)
        
        # Keep raw_captions as numpy array
        if isinstance(raw_captions, list):
            data['raw_captions'] = np.array(raw_captions)
        
        # Save updated metadata
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   {split}:")
        print(f"      captions_input shape: {captions_input.shape}")
        print(f"      captions_target shape: {captions_target.shape}")
        print(f"      Saved to {filepath}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Preprocessing complete! Both approaches can now use the data.")
    print("="*70)
    
    # Verification: Show a sample
    print("\n5. Sample verification:")
    with open(metadata_files['train'], 'rb') as f:
        train_data = pickle.load(f)
    
    sample_idx = 0
    raw = train_data['raw_captions'][sample_idx]
    input_seq = train_data['captions_input'][sample_idx]
    target_seq = train_data['captions_target'][sample_idx]
    
    print(f"   Raw caption: '{raw}'")
    print(f"   Input sequence (first 15): {input_seq[:15]}")
    print(f"   Target sequence (first 15): {target_seq[:15]}")
    
    # Decode back to verify
    decoded_input = tokenizer.sequence_to_caption(input_seq)
    decoded_target = tokenizer.sequence_to_caption(target_seq)
    
    print(f"   Decoded input: '{decoded_input}'")
    print(f"   Decoded target: '{decoded_target}'")
    
    return tokenizer


if __name__ == "__main__":
    try:
        tokenizer = complete_preprocessing()
        print("\n[SUCCESS]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

