CUDA_VISIBLE_DEVICES=1 python train.py \
                --batch_size 128 \
                --task nsmc \
                --train_data_path data/nsmc/ratings_train.txt \
                --val_data_path data/nsmc/ratings_test.txt \
                --gpus 1 \
                --seq_len 64 \
                --max_epochs 10



