CUDA_VISIBLE_DEVICES=1 python train.py \
                --task nsmc \
                --test_data_path data/nsmc/ratings_test.txt \
                --do_test \
                --gpus 1 \
                --seq_len 64 \
                --checkpoint_path lightning_logs/version_2/checkpoints/epoch=3-val_acc=0.9250.ckpt



