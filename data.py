import os

import numpy as np
from tqdm import trange
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

from utils import logout, plot_result

def load_glove(word_dict, dim = 100):
    if dim == 100:
        path = '/data/pretrained/Glove/glove.6B.100d.txt'
    elif dim == 300:
        path = '/data/pretrained/Glove/glove.840B.300d.txt'
    word_emb = []
    word_emb = torch.zeros((len(word_dict), dim), dtype = torch.float)
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(' ') # [word emb1 emb2 ... emb n]
            word = data[0]
            if word in word_dict:
                word_emb[word_dict[word]] = torch.tensor([float(i) for i in data[1:]])
    return word_emb

def make_vocab(word_set, variety_set, category_set):
    word_list = list(word_set)
    word_list.insert(0, '[PAD]')
    word_list.insert(1, '[UNK]')
    word_dict = {word_list[i]: i for i in range(len(word_list))}
    word_emb = load_glove(word_dict, 300)
    logout('Load Glove Word embedding: {}'.format(word_emb.shape))

    variety_list = list(variety_set)
    variety_list.insert(0, '<PAD>')
    variety_list.insert(1, '<OOV>')
    variety_dict = {variety_list[i]: i for i in range(len(variety_list))}

    category_list = list(category_set)
    category_list.insert(0, '[PAD]')
    # category_list.insert(1, '<OOV>')
    category_dict = {category_list[i]: i for i in range(len(category_list))}

    tag_list = ['O'] + [i + '-A' for i in ['B', 'I', 'E', 'S']] + [i + '-O' for i in ['B', 'I', 'E', 'S']]
    tag_dict = {tag_list[i]: i for i in range(len(tag_list))}

    vocabs = {
        'word_list': word_list,
        'word_dict': word_dict,
        'word_emb': word_emb,
        'tag_list': tag_list,
        'tag_dict': tag_dict,
        'category_list': category_list,
        'category_dict': category_dict,
        'polarity_list': ['NEU', 'POS', 'NEG'],
        'polarity_dict': {'NEU': 0, 'POS': 1, 'NEG': 2},
        # 'variety_list': variety_list,
        # 'variety_dict': variety_dict,
        'tokenizer': BertTokenizer.from_pretrained('/data/pretrained/bert-base-chinese/')
    }
    return vocabs

def get_comment_data():
    dataset = []
    word_set = set()
    variety_set = set()
    category_set = set()
    with open('./Dataset/comment_labeled.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            comment_id = data['comment_id']
            comment_variety = data['comment_variety'] # 花生
            user_star = data['user_star'] # 2
            comment_text = data['comment_text']
            comment_units = data['comment_units'] # 多个四元组
            dataset.append({
                # 'id': comment_id, 
                'variety': comment_variety, 
                'user_star': user_star, 
                'tokens': comment_text, 
                'comment': comment_units, 
            })
            word_set |= set(comment_text)
            variety_set.add(comment_variety)
            category_set |= {i['aspect'] for i in comment_units}
    vocabs = make_vocab(word_set, variety_set, category_set)
    
    logout('Load dataset: {}'.format(len(dataset)))
    logout('words: {}, variety: {}, category: {}'.format(len(word_set), len(variety_set), len(category_set)))
    return dataset, vocabs

# JD Comment -> BIOES, aspect-opinion co-tagging
def tag_comment(text, aspects, opinions):
    tags = ['O'] * len(text)
    for t in sorted(aspects, key = lambda x: x['tail'] - x['head']):
        if t['tail'] - t['head'] == 1:
            tags[t['head']] = 'S-A'
        else:
            tags[t['head']] = 'B-A'
            for i in range(t['head'] + 1, t['tail'] - 1):
                tags[i] = 'I-A'
            tags[t['tail'] - 1] = 'E-A'
    for o in sorted(opinions, key = lambda x: x['tail'] - x['head']):
        if o['tail'] - o['head'] == 1:
            tags[o['head']] = 'S-O'
        else:
            tags[o['head']] = 'B-O'
            for i in range(o['head'] + 1, o['tail'] - 1):
                tags[i] = 'I-O'
            tags[o['tail'] - 1] = 'E-O'
    return tags



# ner dataset
class NERDataset(Dataset):
    def __init__(self, args, dataset, vocabs):
        super().__init__()
        self.args = args
        self.batch_size = args['batch_size']
        self.sen_len = args['max_sen_len']

        self.dataset = {
            'tokens': [],
            'token_ids': [],
            'token_masks': [],
            'ner_tag_ids': [],
        }
        self.tokenizer = vocabs['tokenizer']

        tag_dict = vocabs['tag_dict']
        for item in dataset:
            aspects = []
            opinions = []
            for comment in item['comment']: # 每个 comment 是一个四元组，描述一个方面
                aspects += comment['target']
                opinions += comment['opinion']
            tokens = item['tokens']
            ner_tags = tag_comment(tokens, aspects, opinions) + ['O'] * (self.sen_len - len(tokens))
            ner_tag_ids = [tag_dict[i] for i in ner_tags]
            
            bert_tokens = self.tokenizer(tokens, padding = 'max_length', truncation = True, max_length = self.sen_len)
            token_ids = bert_tokens['input_ids']
            token_masks = bert_tokens['attention_mask']
            
            self.dataset['tokens'].append(tokens)
            self.dataset['token_ids'].append(token_ids)
            self.dataset['token_masks'].append(token_masks)
            self.dataset['ner_tag_ids'].append(ner_tag_ids)
        
        self.dataset['token_ids'] = torch.tensor(self.dataset['token_ids'], dtype = int)
        self.dataset['token_masks'] = torch.tensor(self.dataset['token_masks'], dtype = int)
        self.dataset['ner_tag_ids'] = torch.tensor(self.dataset['ner_tag_ids'], dtype = int)
    
    def __len__(self):
        return int(np.ceil(len(self.dataset['token_ids']) / self.batch_size))
    
    def __getitem__(self, index):
        return self.get_batch(index)

    def get_batch(self, index):
        begin = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.dataset['token_ids']))
        
        # cut off paddings
        token_masks = self.dataset['token_masks'][begin: end]
        sen_len = torch.max(torch.sum(token_masks, dim = 1))
        token_masks = token_masks[:, :sen_len].cuda()
        
        token_ids = self.dataset['token_ids'][begin: end, :sen_len].cuda()
        ner_tag_ids = self.dataset['ner_tag_ids'][begin: end, :sen_len].cuda()
        return token_ids, token_masks, ner_tag_ids

def load_ner_dataset(args):
    dataset, vocabs = get_comment_data()
    n_train, n_dev = int(0.6 * len(dataset)), int(0.2 * len(dataset))
    train_set = NERDataset(args, dataset[:n_train], vocabs)
    dev_set = NERDataset(args, dataset[n_train: n_train + n_dev], vocabs)
    test_set = NERDataset(args, dataset[n_train + n_dev:], vocabs)
    return train_set, dev_set, test_set, vocabs

