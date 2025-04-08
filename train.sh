python3 train.py \
    --model 'TRSNet' \
    --dataset 'datasets/DUTS/Train' \
    --lr 0.05 \
    --decay 1e-4 \
    --momen 0.9 \
    --batchsize 10 \
    --loss 'CPR' \
    --savepath 'checkpoint/TRSNet/' \
    --valid True 

