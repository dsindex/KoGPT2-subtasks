CUDA_VISIBLE_DEVICES=1 python train.py \
                --task nsmc \
                --test_data_path data/nsmc/ratings_test.txt \
                --do_test \
                --gpus 1 \
                --seq_len 64 \
                --checkpoint_path lightning_logs/version_0/checkpoints/epoch\=2-step\=3515.ckpt



