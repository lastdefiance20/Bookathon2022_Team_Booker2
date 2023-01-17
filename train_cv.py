import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data_loader_cv import TextClassificationDataset, TextClassificationCollator
from utils_cv_ import read_text, Model, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar, ProgressBarBase
from tqdm import tqdm
import sys

import wandb

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--name', type=str, required=True)
    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_data_name', default='train_fold') # train_data
    p.add_argument('--pretrained_model_name', type=str, default="skt/ko-gpt-trinity-1.2B-v0.5")
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=0)
    p.add_argument('--adam_epsilon', type=float, default=1e-5)
    p.add_argument('--valid_fold', type=int, default=0)
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--decoder_lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=.01)
    p.add_argument('--betas', type=float, default=(0.9, 0.999))
    p.add_argument('--max_grad_norm', type=float, default=1000)
    p.add_argument('--iteration_per_update', type=int, default=1)

    p.add_argument('--cosine', action='store_true')

    p.add_argument('--unfreeze', action='store_true')
    p.add_argument('--additional_training', action='store_true')

    config = p.parse_args()

    return config

def get_loaders(fn, tokenizer, valid_fold, config):

    train_texts, valid_texts = read_text(fn, valid_fold)

    train_loader = DataLoader(
        TextClassificationDataset(train_texts),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    valid_loader = DataLoader(
        TextClassificationDataset(valid_texts),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader
           # index_to_label



def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # num_added_toks = tokenizer.add_tokens(['＇'])
    config.tokenizer = tokenizer

    train_loader, valid_loader = get_loaders(
        config.train_data_name,
        tokenizer,
        config.valid_fold,
        config
    )
    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = int(len(train_loader) / config.iteration_per_update) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    config.n_total_iterations = n_total_iterations
    config.n_warmup_steps = n_warmup_steps

    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    model = Model(config)

    wandb_logger = WandbLogger(project='AIxBookathon', name='{}({}fold)'.format(config.name, config.valid_fold))
    wandb_logger.experiment.config.update(config)

    class LitProgressBar(TQDMProgressBar):

        def __init__(self):
            super().__init__()  # don't forget this :)

        def init_validation_tqdm(self):
            """Override this to customize the tqdm bar for validation."""
            # The main progress bar doesn't exist in `trainer.validate()`
            has_main_bar = self.trainer.state.fn != "validate"
            bar = tqdm(
                desc=self.validation_description,
                position=(2 * self.process_position + has_main_bar),
                disable=True,
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout,
            )
            return bar

    bar = LitProgressBar()

    trainer = Trainer(max_epochs=config.n_epochs,
                      logger=wandb_logger,
                      gpus=1,
                      accelerator='gpu',
                      enable_checkpointing=False,
                      callbacks=[bar]
                      )

    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    seed_everything(seed=42)
    main(config)