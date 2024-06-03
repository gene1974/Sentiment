import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import itertools
import json
import numpy as np
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
        'polarity_list': ['NEU', 'NEG', 'POS'],
        'polarity_dict': {'NEU': 0, 'NEG': 1, 'POS': 2},
        # 'polarity_list': list(range(1, 6)),
        # 'polarity_dict': {i: i for i in list(range(1, 6))},
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

def sentence_polarity(polaritys):
    if len(polaritys) == 0:
        return 0
    elif len(polaritys) == 1:
        return 0
    else:
        return np.mean(polaritys)

def create_quads(args, datas, vocabs):
    sen_len = args['max_sen_len']
    dataset = {
        'tokens': [],
        'token_ids': [],
        'token_masks': [],
        'ner_tag_ids': [],
        'sen_polarity': [], # sentence_level
        'quad_index': [], # [[head, tail], [], ..]
    }
    quads = {
        'index': [], # 第几句
        'aspects': [],
        'opinions': [],
        'category': [],
        'polarity': [],
    }
    tokenizer = vocabs['tokenizer']

    # quads
    tag_dict = vocabs['tag_dict']
    category_dict = vocabs['category_dict']
    polarity_dict = vocabs['polarity_dict']
    for idx, item in enumerate(datas):
        tokens = item['tokens']
        bert_tokens = tokenizer(tokens, padding = 'max_length', truncation = True, max_length = sen_len)
        token_ids = bert_tokens['input_ids']
        token_masks = bert_tokens['attention_mask']
        
        aspects = [] # ner
        opinions = [] # ner
        polaritys = [] # sentence level SA
        
        dataset['quad_index'].append([len(quads), len(quads) + len(item['comment'])])
        for comment in item['comment']: # 每个 comment 是一个四元组，描述一个方面
            category = category_dict[comment['aspect']]
            polarity = polarity_dict[comment['polarity']]
            for aspect, opinion in itertools.product(comment['target'], comment['opinion']):
                quads['aspects'].append([aspect['head'], aspect['tail']])
                quads['opinions'].append([opinion['head'], opinion['tail']])
                quads['category'].append(category)
                quads['polarity'].append(polarity)
                quads['index'].append(idx)

            aspects += comment['target']
            opinions += comment['opinion']
            polaritys.append(polarity)
        ner_tags = tag_comment(tokens, aspects, opinions) + ['O'] * (sen_len - len(tokens))
        ner_tag_ids = [tag_dict[i] for i in ner_tags]
        sen_polarity = sentence_polarity(polaritys)
        # sen_polarity = item['user_star']
        
        dataset['token_ids'].append(token_ids)
        dataset['token_masks'].append(token_masks)
        dataset['ner_tag_ids'].append(ner_tag_ids)
        dataset['sen_polarity'].append(sen_polarity)
    
    dataset['token_ids'] = torch.tensor(dataset['token_ids'], dtype = int)
    dataset['token_masks'] = torch.tensor(dataset['token_masks'], dtype = int)
    dataset['ner_tag_ids'] = torch.tensor(dataset['ner_tag_ids'], dtype = int)
    dataset['sen_polarity'] = torch.tensor(dataset['sen_polarity'], dtype = int)
    dataset['quad_index'] = torch.tensor(dataset['quad_index'], dtype = int) # (n_dataset, 2)
    quads['aspects'] = torch.tensor(quads['aspects'], dtype = int)
    quads['opinions'] = torch.tensor(quads['opinions'], dtype = int)
    quads['category'] = torch.tensor(quads['category'], dtype = int)
    quads['polarity'] = torch.tensor(quads['polarity'], dtype = int)
    quads['index'] = torch.tensor(quads['index'], dtype = int)
    return dataset, quads

class CommentDataset(Dataset):
    def __init__(self, args, datas, vocabs):
        super().__init__()
        self.args = args
        self.batch_size = args['batch_size']
        self.dataset, self.quads = create_quads(args, datas, vocabs)

    def __len__(self):
        return int(np.ceil(len(self.dataset['token_ids']) / self.batch_size))
    
    def __getitem__(self, index):
        return self.get_batch(index)

    def get_batch(self, index):
        begin = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.dataset['token_ids']))
        
        token_masks = self.dataset['token_masks'][begin: end]
        sen_len = torch.max(torch.sum(token_masks, dim = 1))
        
        token_ids = self.dataset['token_ids'][begin: end, :sen_len].cuda()
        token_masks = token_masks[:, :sen_len].cuda()
        ner_tag_ids = self.dataset['ner_tag_ids'][begin: end, :sen_len].cuda()
        sen_polarity = self.dataset['sen_polarity'][begin: end].cuda()
        quad_index = self.dataset['quad_index'][begin: end].cuda() # 每句对应的 Quad 的范围(是否需要？)

        quad_begin = self.dataset['quad_index'][begin][0]
        quad_end = self.dataset['quad_index'][end][1]
        
        quad_aspects = self.quads['aspects'][quad_begin: quad_end].cuda()
        quad_opinions = self.quads['opinions'][quad_begin: quad_end].cuda()
        quad_category = self.quads['category'][quad_begin: quad_end].cuda()
        quad_polarity = self.quads['polarity'][quad_begin: quad_end].cuda()
        quad_sentence = (self.quads['index'][quad_begin: quad_end] - begin).cuda() # quad 对应的句子

        return token_ids, token_masks, ner_tag_ids, sen_polarity, quad_index, \
               quad_aspects, quad_opinions, quad_category, quad_polarity, quad_sentence

def load_dataset(args):
    dataset, vocabs = get_comment_data()
    n_train, n_dev = int(0.6 * len(dataset)), int(0.2 * len(dataset))
    train_set = CommentDataset(args, dataset[:n_train], vocabs)
    dev_set = CommentDataset(args, dataset[n_train: n_train + n_dev], vocabs)
    test_set = CommentDataset(args, dataset[n_train + n_dev:], vocabs)
    print(len(train_set), len(dev_set), len(test_set))
    return train_set, dev_set, test_set, vocabs

def vocab_args(args, vocabs):
    args['num_ner_tag'] = len(vocabs['tag_dict'])
    args['num_category'] = len(vocabs['category_dict'])
    args['num_polarity'] = len(vocabs['polarity_dict'])
    return args

if __name__ == '__main__':
    # tokens = '要使用预训练的模型，我们需要将输入数据转换成合适的格式，以便每个句子都可以发送到预训练的模型中，从而获得相应的嵌入。'
    # tokenizer = BertTokenizer.from_pretrained('/data/pretrained/bert-base-chinese/')
    # bert_tokens = tokenizer(tokens, padding = 'max_length', truncation = True, max_length = 40)
    # print(bert_tokens)
    # print(len(bert_tokens['input_ids']))

    with open('./Dataset/comment_labeled.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            comment_id = data['comment_id']
            comment_variety = data['comment_variety'] # 花生
            user_star = data['user_star'] # 2
            comment_text = data['comment_text']
            comment_units = data['comment_units'] # 多个四元组
            for comment in comment_units:
                print(comment)
            break
    
    # args = {
    #     'batch_size': 8,
    #     'max_sen_len': 40,
    #     'ner_model': 'bert',
    # }
    # train_set, dev_set, test_set, vocabs = load_dataset(args)
    # for i in range(len(train_set)):
    #     try:
    #         batch = train_set.get_batch(i)
    #     except:
    #         print('err: ', i)
    #         continue
    #     # break
    # print(i)

