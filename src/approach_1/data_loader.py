# Data Loader for CNN+LSTM Approach
# Streams images on-the-fly to avoid memory overload

import numpy as np
import pickle
from PIL import Image
from pathlib import Path
import sys

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensorflow.keras.utils import Sequence
from src.helpers.tokenizer import CaptionTokenizer


class ArtEmisDataGenerator(Sequence):
    # Keras Sequence interface for efficient data loading
    # Loads one batch at a time during training
    
    def __init__(self, metadata_path, tokenizer_path, image_base_dir, 
                 batch_size=32, image_size=(128, 128), shuffle=True, **kwargs):
        super().__init__(**kwargs)
        
        self.metadata_path = metadata_path
        self.image_base_dir = Path(image_base_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        
        # Load preprocessed metadata (paths + tokenized captions)
        with open(metadata_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Load tokenizer
        self.tokenizer = CaptionTokenizer.load(tokenizer_path)
        
        # Extract data arrays
        self.image_paths = self.data['image_paths']
        self.captions_input = self.data['captions_input']  # With <start>
        self.captions_target = self.data['captions_target']  # With <end>
        self.raw_captions = self.data['raw_captions']
        
        # Create indices for shuffling
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, batch_idx):
        # Get one batch of data
        # Keras calls this automatically during training
        
        # Calculate which samples go in this batch
        start_idx = batch_idx * self.batch_size
        end_idx = (batch_idx + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get paths and captions for this batch
        batch_image_paths = self.image_paths[batch_indices]
        batch_captions_input = self.captions_input[batch_indices]
        batch_captions_target = self.captions_target[batch_indices]
        
        # Load images (this is where the magic happens - only load what we need)
        images = []
        valid_indices = []
        
        for i, img_path in enumerate(batch_image_paths):
            try:
                img = self._load_image(img_path)
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                # Skip corrupt images without crashing
                print(f"Warning: Skipping {img_path}: {e}")
                continue
        
        # Handle edge case: all images in batch were corrupt
        if len(images) == 0:
            print(f"Warning: Empty batch #{batch_idx}, all images failed to load")
            # Return a minimal valid batch instead of recursing
            # This prevents infinite recursion if many batches fail
            dummy_images = np.zeros((1, *self.image_size, 3), dtype=np.float32)
            dummy_input = np.zeros((1, self.tokenizer.max_length), dtype=np.int32)
            dummy_target = np.zeros((1, self.tokenizer.max_length), dtype=np.int32)
            return {'image': dummy_images, 'caption': dummy_input}, dummy_target
        
        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32)
        batch_captions_input = batch_captions_input[valid_indices]
        batch_captions_target = batch_captions_target[valid_indices]
        
        # Return format: (inputs_dict, targets)
        # Using dict format for better Keras compatibility
        return {'image': images, 'caption': batch_captions_input}, batch_captions_target
    
    def _load_image(self, image_path):
        # Load and preprocess a single image
        
        full_path = self.image_base_dir / image_path
        
        # Open image and convert to RGB (handles grayscale)
        img = Image.open(full_path).convert('RGB')
        
        # Resize to target size (128x128 for memory efficiency)
        img = img.resize(self.image_size, Image.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    
    def on_epoch_end(self):
        # Called by Keras at the end of each epoch
        # Re-shuffle data for next epoch
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_generators(data_dir='data/processed', image_dir='.', 
                    batch_size=32, image_size=(128, 128)):
    # Helper function to create train/val/test generators
    # Note: image_dir is '.' because metadata contains full paths from project root
    
    data_dir = Path(data_dir)
    tokenizer_path = data_dir / 'tokenizer.pkl'
    
    print("Loading data generators...")
    
    # Training generator (with shuffling)
    train_gen = ArtEmisDataGenerator(
        metadata_path=data_dir / 'train_data_metadata.pkl',
        tokenizer_path=tokenizer_path,
        image_base_dir=image_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True
    )
    
    # Validation generator (no shuffling)
    val_gen = ArtEmisDataGenerator(
        metadata_path=data_dir / 'val_data_metadata.pkl',
        tokenizer_path=tokenizer_path,
        image_base_dir=image_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    # Test generator (no shuffling)
    test_gen = ArtEmisDataGenerator(
        metadata_path=data_dir / 'test_data_metadata.pkl',
        tokenizer_path=tokenizer_path,
        image_base_dir=image_dir,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    print(f"  Train: {len(train_gen.image_paths)} samples, {len(train_gen)} batches/epoch")
    print(f"  Val:   {len(val_gen.image_paths)} samples, {len(val_gen)} batches/epoch")
    print(f"  Test:  {len(test_gen.image_paths)} samples, {len(test_gen)} batches/epoch")
    
    return train_gen, val_gen, test_gen, train_gen.tokenizer


# Test the data loader
if __name__ == "__main__":
    print("="*70)
    print("Testing ArtEmis Data Generator")
    print("="*70)
    
    try:
        train_gen, val_gen, test_gen, tokenizer = load_generators(batch_size=16)
        
        print("\nFetching first batch from train generator...")
        inputs, captions_target = train_gen[0]
        images = inputs['image']
        captions_input = inputs['caption']
        
        print(f"  Images shape: {images.shape}")
        print(f"  Captions input shape: {captions_input.shape}")
        print(f"  Captions target shape: {captions_target.shape}")
        print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Decode a sample caption
        sample_input = tokenizer.sequence_to_caption(captions_input[0])
        sample_target = tokenizer.sequence_to_caption(captions_target[0])
        
        print(f"\n  Sample input:  '{sample_input}'")
        print(f"  Sample target: '{sample_target}'")
        
        print("\n" + "="*70)
        print("[SUCCESS] Data generator working correctly!")
        print("="*70)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()

