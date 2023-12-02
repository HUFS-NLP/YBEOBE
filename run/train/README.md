### How to run
'''
python -m train train \
    --output-dir /YBEOBE/models/ \
    --seed 42 --epoch 30 \
    --learning-rate 4e-5 --weight-decay 0.008 \
    --batch-size 64 --valid-batch-size 64 \
    --model-choice LSTM_attention
'''
