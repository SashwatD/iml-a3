# Approach 1.2 - Quick Start Guide

## 🚀 **What This Is**

An **improved version** of the baseline CNN+LSTM model with 7 key enhancements targeting **2-3x better accuracy**.

**Current Baseline:** 9.35% accuracy  
**Target Improved:** 20-30% accuracy

---

## ⚡ **Quick Start (3 Steps)**

### **Step 1: Run Training**

Double-click or run:
```bash
src\approach_1.2\run_improved_training.bat
```

**OR** manually:
```bash
.venv\Scripts\activate
python src\approach_1.2\train.py --embedding learned --epochs 30 --batch_size 16
```

### **Step 2: Monitor Progress**

Open: `models/approach_1.2/improved_cnn_lstm_learned_training_log.csv`

Watch these metrics every few epochs:
- **Val accuracy:** Should reach 20-30% (baseline was 9%)
- **Val loss:** Should drop below 4.0 (baseline was 4.36)
- **Gap (train - val loss):** Should be 0.3-0.4 (baseline was 0.73)

### **Step 3: Evaluate Results**

After training completes:
```bash
python src\approach_1\evaluate.py --model improved --compare_baseline
```

---

## 📊 **What Changed?**

| **Improvement** | **Baseline** | **Improved** | **Expected Gain** |
|----------------|-------------|-------------|-------------------|
| CNN Depth | 4 layers | 13 layers | +5-8% |
| Model Size | 256 units | 512 units | +3-5% |
| Dropout | 0.3 | 0.5 | Reduces overfitting |
| Image Size | 128×128 | 224×224 | +2-3% |
| Grad Clipping | ❌ | ✅ clipnorm=1.0 | Stability |
| LR Schedule | ❌ | ✅ Warmup + Decay | +1-2% |
| Augmentation | ❌ | ✅ Flip, Brightness | Reduces overfitting |

**Total Expected:** +10-20% absolute accuracy improvement

---

## 🖥️ **System Requirements**

### **Minimum:**
- GPU: RTX 3060 (8GB) or equivalent
- RAM: 16GB
- Time: ~2.5-3 hours

### **If you have less GPU:**
```bash
python src\approach_1.2\train.py --batch_size 8  # Reduce from 16 to 8
```

---

## 📁 **Files Created**

```
src/approach_1.2/
├── cnn/cnn_encoder_improved.py          # VGG-style CNN (13 layers, 512 filters)
├── lstm/lstm_decoder_improved.py        # 512-unit LSTM with 0.5 dropout
├── caption_model.py                     # Combined model with improvements
├── data_loader.py                       # Supports 224×224 images + augmentation
├── train.py                             # Training with all optimizations
├── run_improved_training.bat            # ← RUN THIS to start training
├── DETAILED_EXPLANATION.md              # ← READ THIS for viva preparation
├── README.md                            # Comprehensive documentation
└── QUICK_START.md                       # ← YOU ARE HERE
```

---

## 🎓 **For Your Viva**

**READ:** `DETAILED_EXPLANATION.md`

It contains:
- Why each improvement was made
- Mathematical/theoretical justification
- Expected vs actual results
- Common viva questions with detailed answers
- Key papers to cite

**Key talking points:**
1. "We identified 7 specific weaknesses in the baseline"
2. "Each improvement is backed by seminal papers (VGG, Dropout, etc.)"
3. "Expected 2-3x accuracy improvement: from 9% to 20-30%"
4. "All improvements are within assignment scope (no pre-training)"

---

## ⏱️ **Timeline**

| **Time** | **Status** | **Metrics** |
|---------|-----------|-------------|
| 0 min | Training starts | Loss: ~6.0, Acc: ~3% |
| 30 min | Warmup complete (epoch 5) | Loss: ~5.0, Acc: ~7% |
| 90 min | Main learning (epoch 15) | Loss: ~4.0, Acc: ~18% |
| 150 min | Convergence (epoch 25) | Loss: ~3.5, Acc: ~25% |
| 180 min | **DONE** (epoch 30) | **Target: 20-30% acc** |

---

## 🔍 **Troubleshooting**

### **"Out of Memory"**
```bash
python src\approach_1.2\train.py --batch_size 8
```

### **"Accuracy stuck at 5%"**
- Check data loader: Are images loading correctly?
- Verify tokenizer: Is vocabulary size 5000?
- Check normalization: Images should be [0, 1]

### **"Training too slow (>10 min/epoch)"**
- Normal is 5-6 min/epoch with 224×224 images
- Check GPU utilization (should be >80%)
- Close other GPU-heavy programs

### **"Better than baseline but not 20-30%"**
- 15-18% is still good! (1.5-2x improvement)
- Try training longer (40-50 epochs)
- Ensure data augmentation is ON

---

## 📈 **Success Criteria**

**Minimum Success:** 15%+ accuracy (1.5x better than baseline)  
**Target Success:** 20-30% accuracy (2-3x better)  
**Exceptional:** 30%+ accuracy (3x better)

**Also check:**
- Overfitting gap < 0.5 (baseline was 0.73)
- Val loss < 4.0 (baseline was 4.36)
- BLEU-1 > 0.35 (baseline was ~0.25)

---

## 🎯 **Next Steps After Training**

1. **Compare Results:**
   - Create side-by-side comparison table
   - Plot training curves (baseline vs improved)
   - Generate sample captions for same images

2. **Try Other Embeddings:**
   ```bash
   python src\approach_1.2\train.py --embedding word2vec --epochs 30 --batch_size 16
   python src\approach_1.2\train.py --embedding glove --epochs 30 --batch_size 16
   ```

3. **Prepare Viva:**
   - Study `DETAILED_EXPLANATION.md`
   - Practice explaining each improvement
   - Be ready to discuss trade-offs

---

## ✅ **Checklist**

Before starting:
- [ ] Virtual environment activated (`.venv\Scripts\activate`)
- [ ] Data preprocessed (`data/processed/*.pkl` files exist)
- [ ] GPU has 8GB+ VRAM (check with `nvidia-smi`)
- [ ] ~3 hours available for training

During training:
- [ ] Monitor GPU usage (should be >80%)
- [ ] Check training log every 5 epochs
- [ ] Ensure accuracy is increasing

After training:
- [ ] Final accuracy: 20-30% ✓
- [ ] Overfitting gap: <0.5 ✓
- [ ] Model files saved in `models/approach_1.2/` ✓
- [ ] Ready to compare with baseline ✓

---

**Ready? Run this now:**

```bash
src\approach_1.2\run_improved_training.bat
```

**Questions?** Check `README.md` or `DETAILED_EXPLANATION.md`

**Good luck! 🚀**

