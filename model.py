import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

class NERModel(nn.Module):
    def __init__(self, args, vocabs):
        super().__init__()
        self.args = args
        self.vocabs = vocabs
        word_emb = vocabs['word_emb']
        tag_dict = vocabs['tag_dict']
        if args['ner_model'] == 'bert':
            self.bert = BertModel.from_pretrained('/data/pretrained/bert-base-chinese/')
            self.cls = nn.Linear(args['bert_out_dim'], args['ner_class'])
        if args['ner_model'] == 'lstm':
            self.word_embedding = nn.Embedding.from_pretrained(word_emb)
            self.lstm = nn.LSTM(word_emb.shape[1], args['lstm_dim'] // 2,
                                num_layers = args['lstm_layers'], bidirectional=True)
            self.cls = nn.Linear(args['lstm_dim'], args['ner_class'])
        
        self.dropout = nn.Dropout(p = args['dropout'])
        if args['ner_use_crf']:
            self.crf = CRF({tag_dict[tag]: tag for tag in tag_dict})
        else:
            self.loss = nn.CrossEntropyLoss()
        
    def _get_lstm_out(self, tokens, masks):
        embeds = self.word_embedding(tokens)
        embeds = self.dropout(embeds) # (batch_size, sen_len, 256)
        sen_len = torch.sum(masks, dim = 1, dtype = torch.int64).to('cpu') # (batch_size)
        pack_seq = pack_padded_sequence(embeds, sen_len, batch_first = True, enforce_sorted = False)
        lstm_out, _ = self.lstm(pack_seq)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
        return lstm_out
    
    def forward(self, tokens, masks, labels):
        # print(tokens.device, masks.device, labels.device)
        if self.args['ner_model'] == 'bert':
            # word_rep = self.bert(tokens)['last_hidden_state'] # (n_batch, n_tokens, n_emb)
            word_rep = self.bert(tokens, masks)['last_hidden_state'] # (n_batch, n_tokens, n_emb)
        if self.args['ner_model'] == 'lstm':
            word_rep = self._get_lstm_out(tokens, masks)
        
        word_rep = self.dropout(word_rep)
        cls_out = self.cls(word_rep) # （batch_size, seq_len, tagset_size)
        # print(cls_out.device, masks.device, labels.device)
        
        if self.args['ner_use_crf']:
            log_likelihood = self.crf(cls_out, labels, masks)
            n_batch = labels.shape[0]
            loss = -log_likelihood / n_batch
        else:
            loss = self.loss(cls_out.permute(0, 2, 1), labels)
        return loss
    
    def predict(self, tokens, masks):
        if self.args['ner_model'] == 'bert':
            word_rep = self.bert(tokens)['last_hidden_state'] # (n_batch, n_tokens, n_emb)
        if self.args['ner_model'] == 'lstm':
            word_rep = self._get_lstm_out(tokens, masks)
        
        word_rep = self.dropout(word_rep)
        cls_out = self.cls(word_rep) # （batch_size, seq_len, tagset_size)
        
        if self.args['ner_use_crf']:
            predict = self.crf.viterbi_tags(cls_out, masks)
        else:
            predict = torch.argmax(cls_out, dim = 2)
        return predict



