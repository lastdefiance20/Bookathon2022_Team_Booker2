import argparse
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils_cv_ import Model
from copy import deepcopy

def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', type=str, default='model_save')
    p.add_argument('--pretrained_model_name', type=str, default="skt/ko-gpt-trinity-1.2B-v0.5")

    p.add_argument('--max_length', type=int, default=512)

    p.add_argument('--fine_tune', action='store_true')  # Whether using find-tuned-parameter vs huggingface model parameter
    p.add_argument('--name', type=str, default=None)

    p.add_argument('--prompt', type=str, default='나는 금년 여섯 살 난 처녀애입니다. 내 이름은 박옥희이구요. 우리 집 식구라고는 세상에서 제일 예쁜 우리 어머니와 단 두 식구뿐이랍니다. 아차 큰일났군, 외삼촌을 빼놓을 뻔했으니.')

    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=.85)
    p.add_argument('--num_return_sequence', type=int, default=5)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--repetition_penalty', type=float, default=1)


    config = p.parse_args()

    return config

def main(config):

    with torch.no_grad():
        if config.fine_tune:   # using find-tuned-parameter
            path = os.path.join(config.model_fn,
                                config.pretrained_model_name,
                                config.name,
                                )

            model_config = torch.load(os.path.join(path, 'best__.pt'), map_location='cuda:0')['config']
            model = Model(model_config).cuda()
            model.load_state_dict(torch.load(os.path.join(path, 'best__.pt'), map_location='cuda:0')['model'])

        else: # huggingface model parameter
            model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name).cuda()


        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        # num_added_toks = tokenizer.add_tokens(["[CLS1]", "[CLS2]", "[CLS3]", "[CLS4]", '＇'])
        # model.model.resize_token_embeddings(len(tokenizer))


        model.eval()

        prompt = torch.tensor(tokenizer([config.prompt])['input_ids']).cuda()  # 내 방 안에는

        output_tensor = model.generate(prompt,
                                       do_sample=True,
                                       top_k=config.top_k,
                                       top_p=config.top_p,
                                       num_return_sequences=config.num_return_sequence,
                                       temperature=config.temperature,                # default in GPT2
                                       max_length=config.max_length,
                                       repetition_penalty=config.repetition_penalty
                                       )     # ( num_return_sequence , max_length )
                                       # no_repeat_ngram_size = 3
        for i in range(config.num_return_sequence):
            output = tokenizer.decode(output_tensor[i], skip_special_tokens=False)
            print(output)


if __name__ == '__main__':
    config = define_argparser()
    main(config)