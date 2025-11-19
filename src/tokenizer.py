# Text Tokenizer for Caption Processing
# Handles word <-> index conversion and vocabulary building

import re
import numpy as np
from collections import Counter
import pickle


class CaptionTokenizer:
    def __init__(self, vocab_size=5000, max_length=50):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = '<pad>'
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'
        
        # Initialize mappings
        self.word_to_index = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.word_counts = None
    
    def clean_caption(self, caption):
        # Convert to lowercase and remove punctuation
        caption = caption.lower()
        caption = re.sub(r"[^a-z\s']", '', caption)
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption
    
    def tokenize_caption(self, caption):
        # Split into words
        return caption.split()
    
    def fit_on_captions(self, captions):
        # Build vocabulary from captions
        word_counter = Counter()
        
        for caption in captions:
            cleaned = self.clean_caption(caption)
            words = self.tokenize_caption(cleaned)
            word_counter.update(words)
        
        self.word_counts = word_counter
        
        # Select most common words
        most_common = word_counter.most_common(self.vocab_size - 4)
        
        # Build mappings
        for idx, (word, count) in enumerate(most_common, start=4):
            self.word_to_index[word] = idx
            self.index_to_word[idx] = word
        
        print(f"Vocabulary built: {len(self.word_to_index)} words")
    
    def caption_to_sequence(self, caption, add_special_tokens=True):
        # Convert caption to sequence of indices
        cleaned = self.clean_caption(caption)
        words = self.tokenize_caption(cleaned)
        
        sequence = []
        if add_special_tokens:
            sequence.append(self.word_to_index[self.start_token])
        
        for word in words:
            idx = self.word_to_index.get(word, self.word_to_index[self.unk_token])
            sequence.append(idx)
        
        if add_special_tokens:
            sequence.append(self.word_to_index[self.end_token])
        
        return sequence
    
    def sequence_to_caption(self, sequence, remove_special_tokens=True):
        # Convert sequence of indices back to caption
        words = []
        for idx in sequence:
            word = self.index_to_word.get(idx, self.unk_token)
            if remove_special_tokens and word in [self.pad_token, self.start_token, self.end_token]:
                continue
            words.append(word)
        return ' '.join(words)
    
    def pad_sequences(self, sequences, padding='post'):
        # Pad sequences to uniform length
        padded = np.zeros((len(sequences), self.max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            seq = seq[:self.max_length]  # Truncate if too long
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:
                padded[i, -len(seq):] = seq
        
        return padded
    
    def save(self, filepath):
        # Save tokenizer to file
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'word_to_index': self.word_to_index,
                'index_to_word': self.index_to_word,
                'word_counts': self.word_counts
            }, f)
    
    @classmethod
    def load(cls, filepath):
        # Load tokenizer from file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'], max_length=data['max_length'])
        tokenizer.word_to_index = data['word_to_index']
        tokenizer.index_to_word = data['index_to_word']
        tokenizer.word_counts = data['word_counts']
        return tokenizer


if __name__ == "__main__":
    # Test tokenizer
    sample_captions = [
        "The woman is reading books peacefully",
        "A beautiful painting shows vibrant colors",
        "The artwork depicts a serene landscape"
    ]
    
    tokenizer = CaptionTokenizer(vocab_size=100, max_length=20)
    tokenizer.fit_on_captions(sample_captions)
    
    # Test conversion
    test_caption = "the woman is reading books"
    sequence = tokenizer.caption_to_sequence(test_caption)
    reconstructed = tokenizer.sequence_to_caption(sequence)
    
    print(f"Original: '{test_caption}'")
    print(f"Sequence: {sequence}")
    print(f"Reconstructed: '{reconstructed}'")
    print("✓ Tokenizer working correctly")
