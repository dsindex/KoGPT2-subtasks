import argparse
import logging

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import ArgsBase, NSMCDataModule, KorSTSDataModule
from model import SubtaskGPT2, SubtaskGPT2Regression, Classification

parser = argparse.ArgumentParser(description='Train KoGPT2 subtask model')

parser.add_argument('--task', type=str, default=None, help='subtask name')
parser.add_argument('--do_test', action='store_true', help='evaluate on test set')
parser.add_argument('--checkpoint_path', type=str, default=None)

if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = Classification.add_model_specific_args(parser)
    parser = NSMCDataModule.add_model_specific_args(parser)
    parser = KorSTSDataModule.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    if args.task.lower() == 'nsmc':
        dm = NSMCDataModule(args.train_data_path,
                            args.val_data_path,
                            args.test_data_path,
                            batch_size=args.batch_size,
                            max_seq_len=args.seq_len,
                            num_workers=args.num_workers)
        args.num_labels = 2
        model = SubtaskGPT2(args)
    elif args.task.lower() == 'korsts':
        dm = KorSTSDataModule(args.train_data_path,
                              args.val_data_path,
                              batch_size=args.batch_size,
                              max_seq_len=args.seq_len,
                              num_workers=args.num_workers)
        args.num_labels = 1
        model = SubtaskGPT2Regression(args)
    else:
        assert False, 'no task matched!'

    checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{val_acc:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            verbose=True)
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    if args.do_test:
        model = SubtaskGPT2.load_from_checkpoint(args.checkpoint_path)
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
