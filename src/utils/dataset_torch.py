import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import collections
import pickle
import re
import string

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Simple tokenizer: lowercase, remove punctuation, split by space
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

            if frequencies[word] == self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        
        # Add remaining words that met threshold
        for word, count in frequencies.items():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]

EMOTION_MAP = {
    "amusement": 0,
    "anger": 1,
    "awe": 2,
    "contentment": 3,
    "disgust": 4,
    "excitement": 5,
    "fear": 6,
    "sadness": 7,
    "something else": 8
}

class ArtemisDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=2, vocab=None, split='train', max_length=50):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform 
        self.max_length = max_length
        
        # Get captions and image paths
        self.df['image_path'] = self.df.apply(lambda x: os.path.join(root_dir, x['art_style'], x['painting'] + '.jpg'), axis=1)
        
        # Filter for existing images
        # This might be slow if there are many files, but necessary if CSV > Images
        print(f"Filtering dataset... Initial size: {len(self.df)}")
        self.df = self.df[self.df['image_path'].apply(os.path.exists)]
        print(f"Filtered dataset size: {len(self.df)}")
        
        self.captions = self.df["utterance"].tolist()
        self.image_files = self.df['image_path'].tolist()
        self.emotions = self.df['emotion'].tolist()

        # Initialize vocabulary and build vocab
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = self.image_files[index]
        emotion = self.emotions[index]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])
        
        # Padding
        if len(numericalized_caption) > self.max_length:
            numericalized_caption = numericalized_caption[:self.max_length]
        else:
            numericalized_caption += [self.vocab.stoi["<pad>"]] * (self.max_length - len(numericalized_caption))
            
        emotion_idx = EMOTION_MAP.get(emotion, EMOTION_MAP["something else"])

        return image, torch.tensor(numericalized_caption), torch.tensor(emotion_idx)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        emotions = torch.tensor([item[2] for item in batch])
        return imgs, targets, emotions

def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    vocab=None,
    max_length=50
):
    dataset = ArtemisDataset(root_folder, annotation_file, transform=transform, vocab=vocab, max_length=max_length)
    pad_idx = dataset.vocab.stoi["<pad>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx=pad_idx),
    )

    return loader, dataset

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab
