import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

'''
allennlp/allennlp/modules/token_embedders/elmo_token_embedder.py 
class ElmoTokenEmbedder(TokenEmbedder):
    Compute a single layer of ELMo representations.
    This class serves as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_output_representations=1, ...)

batch_to_ids: torch: (len(batch), max sentence length, max word length)
'''
class ElmoEmbedding(nn.Module):
    def __init__(self, lm_emb_dim, num_output_representations = 1, requires_grad = False, dropout = 0.1):
        super().__init__()
        path = '/data/ELMo/'
        self.num_output_representations = num_output_representations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fine_tune = False

        self.lm_embeds = Elmo(
            options_file = path + 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json', 
            weight_file = path + 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5', 
            num_output_representations = 1,
            requires_grad = self.fine_tune,
            dropout = dropout
        )
        self.lm_dense = False
        if self.lm_dense:
            self.lm_emb_dim = lm_emb_dim
            self.pretrained_dim = self.lm_embeds.get_output_dim()
            self.lm_dense = nn.Linear(self.pretrained_dim, self.lm_emb_dim)
        else:
            self.lm_emb_dim = self.lm_embeds.get_output_dim()
        
        print('Load Elmo. fine-tune = {}, Size = {}'.format(self.fine_tune, self.lm_emb_dim))

    def get_emb_dim(self):
        return self.lm_emb_dim

    def forward(self, text):
        '''
        input:
            text: list(list)
            word_ids: (batch_size, max_sen_len)
        output:
            word_embeds: (batch_size, max_sen_len, emb_len)
        '''
        char_ids = batch_to_ids(text).to(self.device)
        lm_emb = self.lm_embeds(char_ids)['elmo_representations'] # List[torch.Tensor]
        if self.num_output_representations == 1:
            lm_emb = lm_emb[0]
        if self.lm_dense:
            lm_emb = self.lm_dense(lm_emb)
        return lm_emb

'''
reference:
https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py
https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/elmo_token_embedder.py
https://zhuanlan.zhihu.com/p/53803919
'''
