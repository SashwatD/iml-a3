# Approach 1.2: Improved CNN+LSTM

## Overview

This is an **improved version** of Approach 1 (baseline CNN+LSTM) with systematic enhancements to achieve 2-3x better accuracy.

### Performance Comparison

| **Metric** | **Baseline (1.0)** | **Improved (1.2)** | **Improvement** |
|-----------|-------------------|-------------------|-----------------|
| Accuracy | 9.35% | 20-30% (target) | +2-3x |
| Train Loss | 3.82 | 3.0-3.5 (target) | -15-20% |
| Val Loss | 4.36 | 3.5-4.0 (target) | -10-20% |
| Overfitting Gap | 0.54 | 0.3-0.4 (target) | -40% |

---

## What's Improved?

### 1. **Deeper CNN (VGG-Style)**
- **Baseline:** 4 conv layers, 256 max filters
- **Improved:** 13 conv layers, 512 max filters
- **Benefit:** Better hierarchical feature learning (+5-8% accuracy)

### 2. **Larger Capacity**
- **Baseline:** 256D embeddings, 256 LSTM units
- **Improved:** 512D embeddings, 512 LSTM units
- **Benefit:** Better pattern modeling (+3-5% accuracy)

### 3. **Higher Dropout**
- **Baseline:** 0.3 dropout
- **Improved:** 0.5 dropout
- **Benefit:** Reduces overfitting gap by 40%

### 4. **Larger Images**
- **Baseline:** 128×128 pixels
- **Improved:** 224×224 pixels
- **Benefit:** More visual details (+2-3% accuracy)

### 5. **Gradient Clipping**
- **New:** clipnorm=1.0
- **Benefit:** Prevents training collapse, stabilizes LSTM

### 6. **Learning Rate Scheduling**
- **New:** Warmup (5 epochs) + Exponential decay
- **Benefit:** Better convergence (+1-2% accuracy)

### 7. **Data Augmentation**
- **New:** Horizontal flip, brightness, contrast
- **Benefit:** Reduces overfitting, "free" extra data

---

## File Structure

```
src/approach_1.2/
├── cnn/
│   └── cnn_encoder_improved.py      # VGG-style CNN (13 layers)
├── lstm/
│   └── lstm_decoder_improved.py     # Improved LSTM (512 units)
├── caption_model.py                 # Combined model
├── data_loader.py                   # Data generator (224×224 support)
├── train.py                         # Training script
├── run_improved_training.bat        # Batch script to run training
├── DETAILED_EXPLANATION.md          # Technical documentation for viva
└── README.md                        # This file
```

---

## How to Use

### Step 1: Train the Model

**Option A: Using Batch Script (Easiest)**
```bash
cd C:\Users\sashw\iml-a3
src\approach_1.2\run_improved_training.bat
```

**Option B: Using Command Line**
```bash
cd C:\Users\sashw\iml-a3
.venv\Scripts\activate
python src\approach_1.2\train.py --embedding learned --epochs 30 --batch_size 16
```

### Step 2: Monitor Training

Training logs are saved to:
- `models/approach_1.2/improved_cnn_lstm_learned_training_log.csv`

Watch for:
- Accuracy increasing (target: 20-30%)
- Val loss decreasing (target: <4.0)
- Overfitting gap shrinking (target: <0.4)

### Step 3: Evaluate Results

Compare with baseline:
```bash
python src\approach_1\evaluate.py --model_path models/approach_1.2/improved_cnn_lstm_learned_best.h5
```

---

## Training Parameters

### Default Configuration
```python
embedding_dim = 512      # UP from 256
lstm_units = 512         # UP from 256
dropout_rate = 0.5       # UP from 0.3
image_size = (224, 224)  # UP from (128, 128)
batch_size = 16          # DOWN from 32 (larger images)
epochs = 30              # UP from 25 (larger model needs more time)
learning_rate = 0.001    # Same, but with warmup + decay
```

### Embedding Types

Train with different embeddings:

**Learned (default):**
```bash
python src\approach_1.2\train.py --embedding learned --epochs 30 --batch_size 16
```

**TF-IDF:**
```bash
python src\approach_1.2\train.py --embedding tfidf --epochs 30 --batch_size 16
```

**Word2Vec:**
```bash
python src\approach_1.2\train.py --embedding word2vec --epochs 30 --batch_size 16
```

**GloVe:**
```bash
python src\approach_1.2\train.py --embedding glove --epochs 30 --batch_size 16
```

---

## System Requirements

### Minimum Requirements
- **GPU:** RTX 3060 (8GB VRAM) or better
- **RAM:** 16GB
- **Disk:** 2GB free space
- **Time:** 2.5-3 hours per model

### If You Have Less GPU RAM

Reduce batch size:
```bash
python src\approach_1.2\train.py --embedding learned --batch_size 8
```

This will work with 6GB VRAM (may be slower).

---

## Expected Training Time

| **Phase** | **Time** | **What's Happening** |
|----------|---------|---------------------|
| Epochs 0-4 | 30 min | Warmup phase, stabilizing |
| Epochs 5-14 | 60 min | Main learning, accuracy rising |
| Epochs 15-24 | 60 min | Fine-tuning, convergence |
| Epochs 25-30 | 30 min | Final polish, early stop may trigger |
| **Total** | **~2.5-3 hours** | |

---

## Troubleshooting

### "Out of Memory" Error

**Solution 1:** Reduce batch size
```bash
python src\approach_1.2\train.py --batch_size 8
```

**Solution 2:** Use smaller images (not recommended, loses accuracy)
- Edit `train.py`, change `image_size = (224, 224)` to `(128, 128)`

### "Training is very slow"

**Normal:** 5-6 minutes per epoch is expected
- Larger images (224×224)
- Deeper CNN (13 layers)
- Data augmentation

**If slower than 10 min/epoch:**
- Check GPU utilization (should be >80%)
- Close other programs using GPU

### "Accuracy stuck at low value"

**If accuracy <5% after 10 epochs:**
- Check data loader is working (print batch shapes)
- Verify tokenizer loaded correctly
- Ensure images are normalized [0, 1]

**If accuracy plateaus at 15-18%:**
- Normal! May need more epochs or data
- Try reducing dropout to 0.4

---

## Detailed Technical Documentation

For **viva/oral examination**, see:
- **`DETAILED_EXPLANATION.md`** - Complete technical justification

This document includes:
- Why each improvement was made
- Mathematical/theoretical foundations
- Expected vs actual results
- Comparison with baseline
- Common viva questions & answers

---

## Next Steps After Training

1. **Evaluate Performance:**
   ```bash
   python src\approach_1\evaluate.py --model_path models/approach_1.2/improved_cnn_lstm_learned_best.h5
   ```

2. **Generate Sample Captions:**
   - Use the evaluation script to see qualitative improvements
   - Compare captions side-by-side with baseline

3. **Train Other Embedding Types:**
   - TF-IDF, Word2Vec, GloVe
   - Compare which works best

4. **Create Visualizations:**
   - Plot training curves (loss, accuracy)
   - Create comparison tables
   - Include in final report/notebook

---

## Citations for Viva

Key papers used:
1. **VGG:** Simonyan & Zisserman, "Very Deep Convolutional Networks" (ICLR 2015)
2. **Dropout:** Srivastava et al., "Dropout: A Simple Way to Prevent Overfitting" (JMLR 2014)
3. **Gradient Clipping:** Pascanu et al., "On the difficulty of training RNNs" (ICML 2013)
4. **LR Warmup:** Goyal et al., "Accurate, Large Minibatch SGD" (2017)
5. **Image Captioning:** Vinyals et al., "Show and Tell" (CVPR 2015)

---

## Questions?

If training fails or results are unexpected:
1. Check `models/approach_1.2/improved_cnn_lstm_learned_training_log.csv`
2. Review `DETAILED_EXPLANATION.md` for troubleshooting
3. Compare config with baseline in `models/approach_1/`

**Expected final results:**
- Train accuracy: 25-35%
- Val accuracy: 20-30%
- Overfitting gap: 0.3-0.4
- BLEU-1: 0.35-0.45

If you achieve these numbers, your improvement is successful! 🎉

