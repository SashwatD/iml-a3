@echo off
REM Batch script to train improved CNN+LSTM model
REM Run this from the project root directory

echo ======================================================================
echo IMPROVED CNN+LSTM TRAINING (Approach 1.2)
echo ======================================================================
echo.
echo This will train the improved model with:
echo   - VGG-style CNN (13 layers)
echo   - 512 units (embedding + LSTM)
echo   - 0.5 dropout
echo   - 224x224 images
echo   - Gradient clipping
echo   - LR warmup + decay
echo   - Data augmentation
echo.
echo Expected: 20-30%% accuracy (vs baseline 9%%)
echo Training time: ~2.5-3 hours (30 epochs)
echo.
echo ======================================================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate

REM Run training with learned embeddings
python src\approach_1_2\train.py --embedding learned --epochs 30 --batch_size 16 --name improved_cnn_lstm_learned

echo.
echo ======================================================================
echo TRAINING COMPLETE!
echo ======================================================================
echo.
echo Check models/approach_1_2/ for:
echo   - improved_cnn_lstm_learned_best.h5 (best model)
echo   - improved_cnn_lstm_learned_training_log.csv (metrics)
echo   - improved_cnn_lstm_learned_config.json (configuration)
echo.
echo Next steps:
echo   1. Compare with baseline (src/approach_1/evaluate.py)
echo   2. Train with other embeddings (tfidf, word2vec, glove)
echo   3. Generate sample captions
echo.

pause

