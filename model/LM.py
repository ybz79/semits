import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Constants as Constants
import logging
import codecs
import pickle
import numpy as np

logger = logging.getLogger()


class LanguageModel(nn.Module):
    def __init__(self, params, emb_size, hidden_size, ouput_size, dropout=0.1):
        super(LanguageModel, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = ouput_size
        self.dropout = dropout

        self.embedding = nn.Embedding(ouput_size, hidden_size)

        if params.us_pretrain_embedding:
            self.init_embedding(params)

        self.rnn = nn.LSTM(self.emb_size, self.hidden_size, num_layers=2, dropout=self.dropout, batch_first=True)
        self.dropout = nn.Dropout(self.dropout)

        self.proj = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_seq, tgt_seq):
        input_seq = self.dropout(self.embedding(input_seq))
        out, hidden = self.rnn(input_seq, None)

        out = self.proj(self.dropout(out))
        loss_func = nn.CrossEntropyLoss(ignore_index=Constants.PAD)

        out = out.view(-1, out.size(-1))
        tgt_seq = tgt_seq.view(-1)

        loss = loss_func(out, tgt_seq)

        return loss

    def get_ppl_reward(self, input_seq):
        with torch.no_grad():
            tgt_seq = input_seq[:, 1:].contiguous()
            input_seq = input_seq[:, :-1].contiguous()

            mask = tgt_seq != Constants.PAD

            input_seq = self.embedding(input_seq)
            out, hidden = self.rnn(input_seq, None)

            out = self.proj(out)
            log_prob = F.log_softmax(out, dim=-1)

            index = tgt_seq.unsqueeze(-1)
            select_log_prob = torch.gather(log_prob, -1, index).squeeze(-1) * mask.float()
            ppl_reward = torch.exp(select_log_prob.sum(dim=-1) / select_log_prob.size(-1))
        
        return ppl_reward.cpu().numpy()

    def init_embedding(self, args):
        logger.info("init embeddings with pretrained vector")
        pre_train_path = args.embedding_path
        weight_file = codecs.open(pre_train_path, mode='rb')
        emb_weight = pickle.load(weight_file)

        self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=False)

