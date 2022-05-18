import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
assert(torch.cuda.is_available())
import transformers
transformers.logging.set_verbosity_error()

import argparse


from data import load_ner_dataset
from train import train

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type = str, default='train', choices=['train', 'test'],
                        help='option: train, test')
    parser.add_argument('--task', type = str, default='ner', choices=['ner'], help='')
    
    # ner
    parser.add_argument('--ner_model', type = str, default = 'bert', choices = ['lstm', 'bert'], help = 'option: lstm, bert')
    parser.add_argument('--ner_use_crf', type = bool, default = True)
    parser.add_argument('--ner_class', type = int, default = 9)

    # lstm ner
    parser.add_argument('--lstm_dim', type = int, default = 100)
    parser.add_argument('--lstm_layers', type = int, default = 2)
    # bert_ner
    parser.add_argument('--bert_path', type = str, default = '/data/pretrained/bert-base-chinese/')
    parser.add_argument('--bert_out_dim', type = int, default = 768)
    # train
    parser.add_argument('--max_sen_len', type = int, default = 40)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--dropout', type = float, default = 0.2)
    parser.add_argument('--eval', type = bool, default = True)

    args = parser.parse_args().__dict__
    return args

if __name__ == '__main__':
    args = parse_arg()
    print(args)
    train_set, dev_set, test_set, vocabs = load_ner_dataset(args)
    train_loss, dev_loss = train(args, vocabs, train_set, dev_set)
    
    
