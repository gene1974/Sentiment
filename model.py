import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer

from crf import CRF
device = 'cuda'

'''
原有模型：
NER model: 
bert + max_pooling -> span embedding
ltp segment + embedding(random) -> segment embedding
文本特征表示：
没有 ent 的 token 做 max pooling 得到上下文
ent_span; ent_seg; context_span; ent_span; ent_seg;
'''

# 目前是普通 NER model，后期可以考虑按 span-ASTE model
# NER + sentence-level classification
class NERModel(nn.Module):
    def __init__(self, args, vocabs):
        super().__init__()
        self.args = args
        self.vocabs = vocabs
        word_emb = vocabs['word_emb']
        tag_dict = vocabs['ner_tag_dict']
        if args['ner_model'] == 'bert':
            self.bert = BertModel.from_pretrained('/data/pretrained/bert-base-chinese/')
            self.cls = nn.Linear(args['bert_out_dim'], args['num_ner_tag'])
        if args['ner_model'] == 'lstm':
            self.word_embedding = nn.Embedding.from_pretrained(word_emb)
            self.lstm = nn.LSTM(word_emb.shape[1], args['lstm_dim'] // 2,
                                num_layers = args['lstm_layers'], bidirectional=True)
            self.cls = nn.Linear(args['lstm_dim'], args['num_ner_tag'])
        
        self.dropout = nn.Dropout(p = args['dropout'])
        self.sen_polarity_cls = nn.Linear(args['bert_out_dim'], 3)
        if args['ner_use_crf']:
            self.crf = CRF({tag_dict[tag]: tag for tag in tag_dict})
        self.loss = nn.CrossEntropyLoss()
        
    def _get_lstm_out(self, tokens, masks):
        embeds = self.word_embedding(tokens)
        embeds = self.dropout(embeds) # (batch_size, sen_len, 256)
        sen_len = torch.sum(masks, dim = 1, dtype = torch.int64).to('cpu') # (batch_size)
        pack_seq = pack_padded_sequence(embeds, sen_len, batch_first = True, enforce_sorted = False)
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
        return lstm_out

    def _extract_entity_single(self, tags):
        aspects, opinions = [], []
        tag_list = self.vocabs['tag_list']
        tags = [tag_list[i] for i in tags]
        i = 0
        while i < len(tags):
            if tags[i][0] == 'B':
                j = i + 1
                while j < len(tags):
                    if tags[j][0] == 'E':
                        break
                    else:
                        j += 1
                if tags[i][-1] == 'A':
                    aspects.append([i, j])
                elif tags[i][-1] == 'O':
                    opinions.append([i, j])
                i = j + 1
            elif tags[i][0] == 'S':
                if tags[i][-1] == 'A':
                    aspects.append([i, i])
                elif tags[i][-1] == 'O':
                    opinions.append([i, i])
                i += 1
            else:
                i += 1
        return aspects, opinions
    
    def extract_entity(self, tags):
        if tags.dim() == 3:
            aspects, opinions = [], []
            for i in tags:
                aspect, opinion = self._extract_entity_single(i)
                aspects.append(aspect)
                opinions.append(opinion)
        else:
            aspects, opinions = self._extract_entity_single(tags)
        return aspects, opinions
    
    def _forward(self, tokens, masks):
        if self.args['ner_model'] == 'bert':
            bert_out = self.bert(tokens, masks)
            word_rep = bert_out['last_hidden_state'] # (n_batch, n_tokens, n_emb)
            pool_rep = bert_out['pooler_output']
        elif self.args['ner_model'] == 'lstm':
            word_rep = self._get_lstm_out(tokens, masks) # (n_batch, n_tokens, emb_dim)
        word_rep = self.dropout(word_rep)

        sen_polar_out = self.sen_polarity_cls(pool_rep)
        ner_out = self.cls(word_rep) # （batch_size, seq_len, tagset_size)

        return ner_out, sen_polar_out, word_rep, pool_rep

    def forward(self, tokens, masks, labels, sen_polarity):
        ner_out, sen_polar_out, word_rep, pool_rep = self._forward(tokens, masks)
        
        if self.args['ner_use_crf']:
            log_likelihood = self.crf(ner_out, labels, masks)
            n_batch = labels.shape[0]
            loss = -log_likelihood / n_batch
        else:
            loss = self.loss(ner_out.permute(0, 2, 1), labels) #（batch_size, seq_len, tagset_size)
        
        sen_polar_loss = self.loss(sen_polar_out, sen_polarity)
        return loss, word_rep
    
    def predict(self, tokens, masks):
        ner_out, sen_polar_out, word_rep, pool_rep = self._forward(tokens, masks)
        
        sen_polar = torch.argmax(sen_polar_out, dim = 1)
        if self.args['ner_use_crf']:
            ner_tag = self.crf.viterbi_tags(ner_out, masks)
        else:
            ner_tag = torch.argmax(ner_out, dim = 2) #（batch_size, seq_len)
        aspects, opinions = self.extract_entity(ner_tag)
        return ner_tag, sen_polar, aspects, opinions

# not used
class ClsModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.width_embedding = nn.Embedding(40, 100)
        out_dim = args['bert_out_dim'] * 2
        self.category_cls = nn.Linear(out_dim, args['num_category'])
        self.polarity_cls = nn.Linear(out_dim, args['num_polarity'])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pair_rep, category, polarity): # (n_batch, rep_dim)
        cate_out = self.category_cls(pair_rep)
        polar_out = self.polarity_cls(pair_rep)
        cate_loss = self.loss(cate_out, category)
        polar_loss = self.loss(polar_out, polarity)
        return cate_loss, polar_loss

    def predict(self, pair_rep):
        cate_out = self.category_cls(pair_rep)
        cate_out = torch.argmax(cate_out, dim = -1)
        polar_out = self.polarity_cls(pair_rep)
        polar_out = torch.argmax(polar_out, dim = -1)
        return cate_out, polar_out # (n_batch)

class QuadModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.width_embedding = nn.Embedding(40, 100)
        out_dim = args['bert_out_dim'] * 2
        self.valid_cls = nn.Linear(out_dim, 2) # is valid pair
        self.category_cls = nn.Linear(out_dim, args['num_category'])
        self.polarity_cls = nn.Linear(out_dim, args['num_polarity'])

    def create_pairs(self, word_rep, aspects, opinions):
        asp_ids, opi_ids = [], []
        pair_reps = []
        # for aspect, opinion in itertools.product(aspects, opinions):
        for i, j in itertools.product(range(len(aspects)), range(len(opinions))):
            aspect, opinion = aspects[i], opinions[j]
            asp_head, asp_tail = aspect # [head, tail)
            asp_width = asp_tail - asp_head
            asp_rep = torch.max(word_rep[asp_head: asp_tail], dim = 1) # max pooling, 可选头尾拼接
            opi_head, opi_tail = opinion # [head, tail)
            opi_width = opi_tail - opi_head
            opi_rep = torch.max(word_rep[opi_head: opi_tail], dim = 1)
            pair_dist = opi_head - asp_head
            pair_rep = torch.cat((asp_rep, opi_rep), dim = -1) # width_emb, sentence_rep
            pair_reps.append(pair_rep)
            asp_ids.append[i]
            opi_ids.append(j)
        pair_reps = torch.stack(pair_reps)
        asp_ids = torch.tensor(asp_ids)
        opi_ids = torch.tensor(opi_ids)
        return pair_reps, asp_ids, opi_ids

    # for single sentence
    def forward(self, word_rep, aspects, opinions):
        pair_reps, asp_ids, opi_ids = self.create_pairs(word_rep, aspects, opinions)
        valid_out = self.valid_cls(pair_reps)
        cate_out = self.category_cls(pair_reps)
        polar_out = self.polarity_cls(pair_reps)

        valid_index = valid_out[:, 0] > valid_out[:, 1]
        return pair_reps[valid_index], asp_ids[valid_index], opi_ids[valid_index], cate_out[valid_index], polar_out[valid_index]



class Model(nn.Module):
    def __init__(self, args, vocabs):
        super().__init__()
        self.args = args
        self.vocabs = vocabs
        self.ner_model = NERModel(args, vocabs)
        self.quad_model = QuadModel(args)

    def forward(self, batch_data):
        token_ids, token_masks, ner_tag_ids, sen_polarity, quad_index, \
            quad_aspects, quad_opinions, quad_category, quad_polarity, quad_sentence = batch_data
        loss, word_rep = self.ner_model(token_ids, token_masks, ner_tag_ids, sen_polarity)
        
        for i in range(word_rep.shape[0]): # n_batch
            quad_begin, quad_end = quad_index[i].tolist()
            pair_reps, asp_ids, opi_ids = self.quad_model(word_rep, quad_aspects[quad_begin: quad_end], quad_opinions[quad_begin: quad_end])

# ltp.seg 分词得到 span
class SpanASTE(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    tokens = '要使用预训练的模型，我们需要将输入数据转换成合适的格式，以便每个句子都可以发送到预训练的模型中，从而获得相应的嵌入。'
    tokenizer = BertTokenizer.from_pretrained('/data/pretrained/bert-base-chinese/')
    bert = BertModel.from_pretrained('/data/pretrained/bert-base-chinese/')
    bert_tokens = tokenizer([tokens, tokens, tokens], padding = 'max_length', truncation = True, max_length = 40, return_tensors = 'pt')
    print(bert_tokens)
    # bert_ids = bert_tokens['input_ids']
    # bert_out1 = bert(bert_tokens['input_ids'])
    # print(bert_out1)
    # bert_out2 = bert(bert_tokens['input_ids'], bert_tokens['attention_mask'])
    # print(bert_out2[0].shape)
    # bert_out3 = bert(**bert_tokens)
    # print(bert_out3)

    # token2 = tokenizer.decode(bert_tokens['input_ids'], skip_special_tokens = True)
    # token2 = tokenizer.convert_ids_to_tokens(bert_tokens['input_ids'][0], skip_special_tokens = True)
    # print(token2)

