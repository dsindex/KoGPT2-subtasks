# Finetuning GPT2 for classification

- how to
```
$ ./train-nmsc.sh
...
Epoch 3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [10:39<00:00,  2.44it/s, loss=0.280, v_num=0, val_acc=0.908, val_acc_2=0.887, train_loss_step=0.059, train_acc=0.984, train_loss_epoch=0.211]              precision    recall  f1-score   support████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [01:02<00:00,  7.03it/s]

           0     0.8842    0.8948    0.8895     24827
           1     0.8950    0.8845    0.8897     25173

    accuracy                         0.8896     50000
   macro avg     0.8896    0.8896    0.8896     50000
weighted avg     0.8897    0.8896    0.8896     50000

[[22215  2612]
 [ 2908 22265]]
Epoch 3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [10:40<00:00,  2.44it/s, loss=0.309, v_num=0, val_acc=0.925, val_acc_2=0.890, train_loss_step=0.180, train_acc=0.929, train_loss_epoch=0.163]Epoch 3, global step 4687: val_loss reached 0.30906 (best 0.28009), saving model to "/data/private/KoGPT2-subtasks/lightning_logs/version_0/checkpoints/epoch=3-val_acc=0.9250.ckpt" as top 3
...

$ ./test-nmsc.sh
...
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 1562/1563 [01:12<00:00, 22.01it/s] precision recall f1-score support

           0 0.8842 0.8948 0.8895 24827
           1 0.8950 0.8845 0.8897 25173

    accuracy 0.8896 50000
   macro avg 0.8896 0.8896 0.8896 50000
weighted avg 0.8897 0.8896 0.8896 50000

[[22215 2612]
 [ 2908 22265]]
...

```


----
# KoGPT2-subtasks 

## KoGPT2 v2.0 한국어 평가 모듈

설치

```bash
git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-subtasks.git
cd KoGPT2-subtasks
pip install -r requirements 
```

## Subtasks

### NSMC

```bash
# sh run_nsmc.sh
CUDA_VISIBLE_DEVICES=0 python train.py \
                --batch_size 128 \
                --task nsmc \
                --train_data_path data/nsmc/ratings_train.txt \
                --val_data_path data/nsmc/ratings_test.txt \
                --gpus 1 \
                --seq_len 64 \
                --max_epochs 10
```

### KorSTS

```bash
# sh run_korsts.sh
CUDA_VISIBLE_DEVICES=0 python train.py \
                --batch_size 64 \
                --task korsts \
                --train_data_path data/KorNLUDatasets/KorSTS/sts-train.tsv \
                --val_data_path data/KorNLUDatasets/KorSTS/sts-test.tsv \
                --gpus 1 \
                --seq_len 64 \
                --max_epochs 5
```


### KorNLI

*Working*

