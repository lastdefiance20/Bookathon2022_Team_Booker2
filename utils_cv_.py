import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import random
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch_optimizer as custom_optim
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
import wandb
import torch.optim as optim
from pytorch_lightning.callbacks import TQDMProgressBar


class Model(pl.LightningModule):

    def __init__(self, cfg):
        self.cfg = cfg
        self.pretrained_model_name = cfg.pretrained_model_name
        self.config = AutoConfig.from_pretrained(cfg.pretrained_model_name)
        super().__init__()

        if not cfg.additional_training:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_model_name, config=self.config).to('cuda:0')
        else:
            root = os.path.join(cfg.model_fn,
                                cfg.pretrained_model_name,
                                cfg.name,
                                'best_.pt')
            self.model = AutoModelForCausalLM.from_pretrained(root, config=self.config).to('cuda:0')

        self.model.resize_token_embeddings(len(cfg.tokenizer))

        self.model.gradient_checkpointing_enable()

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for i, m in enumerate(self.model.transformer.h):
            # Only un-freeze the last n transformer blocks
            if i >= 12:
                for parameter in m.parameters():
                    parameter.requires_grad = True
        for parameter in self.model.transformer.ln_f.parameters():
            parameter.requires_grad = True
        for parameter in self.model.lm_head.parameters():
            parameter.requires_grad = True

        self.best_loss = np.inf

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, x, mask):

        loss = self.model(x, attention_mask=mask, labels=x, use_cache=False).loss

        return loss

    def generate(self, x, **kwargs):
        output_tensor = self.model.generate(x, **kwargs)
        return output_tensor


    def training_step(self, train_batch, batch_idx):
        x, mask, = train_batch['input_ids'], train_batch['attention_mask']
        x, mask = x.to('cuda:{}'.format(self.cfg.gpu_id)), mask.to('cuda:{}'.format(self.cfg.gpu_id))

        if batch_idx == 0:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        with torch.cuda.amp.autocast(enabled=True):
            loss = self.forward(x, mask)

        self.log('train_loss', loss, prog_bar=True)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True)

        self.scaler.scale(loss).backward()

        if (self.cfg.iteration_per_update == 1) | (batch_idx % self.cfg.iteration_per_update == (self.cfg.iteration_per_update - 1)):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1000)
            optimizer = self.optimizers()
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
            self.lr_schedulers().step()

            self.log('grad', grad_norm, prog_bar=True)

            # if batch_idx % 100 == 1:
            #     print('batch : {} / loss : {} / lr : {} / grad : {}'.format(batch_idx, loss,
            #                                                                 self.lr_schedulers().get_lr()[0],
            #                                                                 grad_norm))

        return {'loss' : loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

        print(f'epoch {self.current_epoch} train loss {avg_loss}')

    def validation_step(self, valid_batch, batch_idx):
        x, mask = valid_batch['input_ids'], valid_batch['attention_mask']
        x, mask = x.to('cuda:{}'.format(self.cfg.gpu_id)), mask.to('cuda:{}'.format(self.cfg.gpu_id))

        loss = self.forward(x, mask)

        return {'val_loss' : loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('val_loss', avg_loss)
        self.log('PPL', torch.exp(avg_loss))

        print('valid_loss : {}  /  PPL :  {} '.format(avg_loss, torch.exp(avg_loss)))

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_model(self.cfg, avg_loss)
            print('saved')

        return {'val_loss' : avg_loss, 'PPL' : torch.exp(avg_loss)}


    def get_optimizer_params(self, decoder_lr, weight_decay):
        if self.cfg.use_radam:
            optimizer = custom_optim.RAdam(self.parameters(), lr=self.cfg.lr)
            return optimizer

        else:
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': decoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': decoder_lr, 'weight_decay': 0.0},
            ]

            return optimizer_parameters

    def configure_optimizers(self):
        optimizer_parameters = self.get_optimizer_params(
                                         decoder_lr=self.cfg.decoder_lr,
                                         weight_decay=self.cfg.weight_decay)

        optimizer = optim.AdamW(optimizer_parameters, eps=self.cfg.adam_epsilon, betas=self.cfg.betas)

        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=self.cfg.n_total_iterations
        ) if not self.cfg.cosine else get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=self.cfg.n_total_iterations,
            num_cycles=.5,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def save_model(self, config, loss):
        path = os.path.join(config.model_fn,
                            config.pretrained_model_name,
                            config.name,
                            )

        if not os.path.isdir(path):
            if 'working' not in os.getcwd():
                os.makedirs(path)
            else:
                pass
        else:
            pass

        torch.save(
            {
                'model': self.state_dict(),
                'config' : config,
                # 'loss' : round(loss, 4)
            }, os.path.join(path, 'best_.pt') if 'working' not in os.getcwd() else 'best_{}_{}.pt'.format(str(round(loss, 4)), config.valid_fold)
        )


def read_text(fn, valid_fold):
    data = os.path.join('data', str(fn) + '.csv')
    data = pd.read_csv(data, encoding='utf-8')

    train_data = data[data['fold'] != valid_fold]
    valid_data = data[data['fold'] == valid_fold]
    del data

    train_texts = train_data['text'].values
    valid_texts = valid_data['text'].values
    del train_data, valid_data

    return train_texts, valid_texts

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


