import torch
import model.Constants as Constants
from sklearn.decomposition import TruncatedSVD
import torch.nn as nn
import logging
import codecs
import pickle
import numpy as np

logger = logging.getLogger()


class STS_model(object):
    def __init__(self, params):
        self.params = params
        self.embedding = None
        self.init_embedding(params)
        self.frc_dic = self.get_frequent_dict(params)
        self.len_V = len(self.frc_dic)

    def init_embedding(self, args):
        logger.info("init embeddings with pretrained vector")
        pre_train_path = args.embedding_path
        weight_file = codecs.open(pre_train_path, mode='rb')
        emb_weight = pickle.load(weight_file)

        self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=True)

    def get_frequent_dict(self, args, a=1e-3):
        Nums = 0.
        frequent_dic = {}
        frc_path = args.frc_path
        frc_file = codecs.open(frc_path, mode='r')
        for w in frc_file.readlines():
            line = w.strip().split()
            word = line[0]
            frc = int(line[1])
            Nums += frc

            frequent_dic[word] = frc

        for key, value in frequent_dic.items():
            frequent_dic[key] = a / (a + value / Nums)

        return frequent_dic

    def get_embedding(self, sent_idx, sent, align):
        sent_idx = torch.Tensor(sent_idx).long()
        if len(align) == 0:
            return np.ones(512)
        with torch.no_grad():
            embedding = self.embedding(sent_idx)
            weight = torch.Tensor([self.frc_dic[w] if w in self.frc_dic else 1 for w in sent.split()])
            weight = weight[align].unsqueeze(-1)
            vec = (weight * embedding).sum(dim=0) / embedding.size(0)

        return vec.numpy()

    def get_reward(self, embeddings):
        batch_size = embeddings.shape[0]
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(embeddings)
        pc = svd.components_
        new_emb = embeddings - embeddings.dot(pc.transpose()).dot(pc)

        index1 = list(range(0, batch_size, 2))
        index2 = list(range(1, batch_size, 2))

        emb1 = new_emb[index1]
        emb2 = new_emb[index2]

        emb_dot = (emb1 * emb2).sum(axis=-1)
        emb1_norm = np.linalg.norm(emb1, ord=2, axis=-1)
        emb2_norm = np.linalg.norm(emb2, ord=2, axis=-1)
        pred_sim = emb_dot / (emb1_norm * emb2_norm + 1e-12)

        return pred_sim
