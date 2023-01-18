# !pip install transformers==4.21.2
# !pip install kss

import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import kss
import argparse
from utils_cv_ import Model, seed_everything
seed_everything(seed=42)

def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', type=str, default='model_save')
    p.add_argument('--pretrained_model_name', type=str, default="skt/ko-gpt-trinity-1.2B-v0.5")

    p.add_argument('--fine_tune', action='store_true')  # Whether using find-tuned-parameter vs huggingface model parameter
    p.add_argument('--name', type=str, default=None)

    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=.85)
    p.add_argument('--num_return_sequence', type=int, default=5)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--repetition_penalty', type=float, default=1.5)

    p.add_argument('--prompt', type=str, default='국경의 긴 터널을 빠져 나오자, 설국이었다.')
    
    p.add_argument('--additional_training', action='store_true')

    p.add_argument('--text_save_folder', type=str, default='./saved_text/')
    p.add_argument('--text_save_name', type=str, default='result.txt')
    p.add_argument('--restore_from_text', type=str, default=None)

    config = p.parse_args()

    return config

def main(config):
    with torch.no_grad():
        if config.fine_tune:  # using find-tuned-parameter
            path = os.path.join(config.model_fn,
                                config.pretrained_model_name,
                                config.name,
                                )

            model_config = torch.load(os.path.join(path, 'best_.pt'), map_location='cuda:0')['config']
            model_config.additional_training = False
            model = Model(model_config).cuda()
            model.load_state_dict(torch.load(os.path.join(path, 'best_.pt'), map_location='cuda:0')['model'])

        else:  # huggingface model parameter
            model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name).cuda()

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        # num_added_toks = tokenizer.add_tokens(["[CLS1]", "[CLS2]", "[CLS3]", "[CLS4]", '＇'])
        # model.model.resize_token_embeddings(len(tokenizer))

        model.eval()

        # initial_prompt = torch.tensor(tokenizer([config.prompt])['input_ids']).cuda()  # 내 방 안에는

        initial_prompt = config.prompt
        prompt_list = kss.split_sentences(initial_prompt)

        if config.restore_from_text != None:
            try:
                with open(config.text_save_folder + config.restore_from_text, 'r') as f:
                    prompt_list = f.read().splitlines()
            except:
                print("there is no file called ",config.restore_from_text)

        while True:
            print('=' * 50)
            for i in prompt_list:
                print(i)

            print("word count: ", len(" ".join(prompt_list)))

            candidate_list = []
            input_list = []
            for num in range(1, 11):  # 1개 참조, 2개 참조, ..., 10개 참조
                local_prompt = ''
                input_prompt = ''

                try:
                    for i in range(num, 0, -1):
                        local_prompt += (prompt_list[-i] + ' ')
                        input_prompt += (prompt_list[-i] + ' ')
                except:
                    break

                local_prompt = torch.tensor(tokenizer([local_prompt])['input_ids']).cuda()
                output_tensor = model.generate(local_prompt,  # 3개 참조
                                               do_sample=True,
                                               top_k=config.top_k,
                                               top_p=config.top_p,
                                               num_return_sequences=config.num_return_sequence,
                                               temperature=config.temperature,  # default in GPT2
                                               max_length=local_prompt.shape[1]+64,
                                               repetition_penalty=config.repetition_penalty
                                               )  # ( num_return_sequence , max_length )
                output_tensor = output_tensor[:, local_prompt.shape[1]:]

                for i in range(config.num_return_sequence):
                    output = tokenizer.decode(output_tensor[i], skip_special_tokens=False)
                    output = kss.split_sentences(output)[0].replace('<unk>', '').replace('\n', '').replace('</d>', '')
                    candidate_list.append(output)

                input_list.append(input_prompt)

            for idx, i in enumerate(candidate_list):
                print(f'{idx}th : {i}')

            command = input('select(r=reset_candidate, d=drop_last_sentence, p=reset_hyperparameter, n=New_prompt, b=Break) : ')
            command = command.strip()

            if command == 'p' or command == 'ㅔ':
                print('RESET hyperparameter')
                config.top_k = int(input('top_k : '))
                config.top_p = float(input('top_p : '))
                config.num_return_sequences = int(input('num_return_sequences : '))
                config.temperature = float(input('temperature : '))
                config.repetition_penalty = float(input('repetition_penalty : '))
                continue

            elif command == 'n' or command == 'ㅜ':
                print('SET NEW PROMPT')
                prompt = input('NEW PROMPT : ')
                prompt_list.append(prompt)

                #save prompt_list text data to result.txt using .joint(' ')
                with open(config.text_save_folder + config.text_save_name, 'w') as f:
                    for item in prompt_list:
                        f.write("%s\n" % item)

                continue

            elif command == 'b' or command == 'ㅠ':
                break

            elif command == 'r' or command == 'ㄱ':
                continue

            elif command == 'd' or command == 'ㅇ':
                prompt_list.pop(-1)
                continue

            else:
                command = int(command)

            new_prompt = candidate_list[command]
            # print(new_prompt)
            prompt_list.append(new_prompt.strip())

            #save prompt_list text data to result.txt using .joint(' ')
            with open(config.text_save_folder + config.text_save_name, 'w') as f:
                for item in prompt_list:
                    f.write("%s\n" % item)

if __name__ == '__main__':
    config = define_argparser()
    main(config)