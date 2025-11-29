@echo off
cd /d C:\Users\sashw\iml-a3
call .venv\Scripts\activate.bat
python src\approach_1\train.py --embedding learned --epochs 25 --batch_size 32 --dropout 0.5 --name cnn_lstm_learned
pause

