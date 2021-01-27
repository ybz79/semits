import torch
import os
from torch.nn import functional as F
import codecs
import model.Constants as Constants
from metrics.STAR import SARIsent
from metrics.FKGL import fkgl_score
import metrics.SARI as sari
from nltk.translate.meteor_score import single_meteor_score
from logging import getLogger
import numpy as np
from nltk.corpus import cmudict
from nltk.translate.bleu_score import corpus_bleu

logger = getLogger()

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def calculate_suff(t_keep, t_del, t_add):
    keep_p = [i / j if j > 0 else 0 for i, j in zip(t_keep[:, 0], t_keep[:, 1])]
    keep_r = [i / j if j > 0 else 0 for i, j in zip(t_keep[:, 0], t_keep[:, 2])]

    keep_f = [2 * i * j / (i + j) if i+j > 0 else 0 for i, j in zip(keep_p, keep_r)]

    del_p = [i / j if j > 0 else 0 for i, j in zip(t_del[:, 0], t_del[:, 1])]
    del_r = [i / j if j > 0 else 0 for i, j in zip(t_del[:, 0], t_del[:, 2])]
    del_f = [2 * i * j / (i + j) if i+j > 0 else 0 for i, j in zip(del_p, del_r)]

    add_p = [i / j if j > 0 else 0 for i, j in zip(t_add[:, 0], t_add[:, 1])]
    add_r = [i / j if j > 0 else 0 for i, j in zip(t_add[:, 0], t_add[:, 2])]
    add_f = [2 * i * j / (i + j) if i+j > 0 else 0 for i, j in zip(add_p, add_r)]

    score = sum(keep_f) / 4 + sum(del_f) / 4 + sum(add_f) / 4
    return score / 3, sum(keep_f) / 4, sum(del_f) / 4, sum(add_f) / 4


class Evaluator(object):

    def __init__(self, model, lm, data, params):
        self.model = model
        self.lm = lm
        self.data = data
        self.index2word = data['index2word']
        self.params = params
        self.type_dict = {'comp': 0, 'simp': 1}
        self.cmu_dict = cmudict.dict()
        self.ref_list = []
        self.output_list = []

    def get_loader(self, loader_name, src_type, tgt_type, mode='dev'):
        if loader_name == 'encdec':
            if tgt_type is None:
                data_loader = self.data['mono'][loader_name][src_type + '_dev']
            else:
                data_loader = self.data['para'][mode]
        else:
            data_loader = self.data['mono']['otf'][src_type + '_dev']
        return data_loader

    def merge_subword(self, subword_list):
        ret_sent = []
        prev = ""
        for word in subword_list:
            if "##" in word:
                prev += word.strip("##")
            else:
                if prev != "":
                    ret_sent.append(prev)
                prev = word

        if prev != "":
            ret_sent.append(prev)

        return " ".join(ret_sent)

    def get_reward(self, input_seq, policy_gen, tgt_seq, type='simp'):

        def normal_fkgl(score, type):
            if type == 'simp':
                min_score, max_score = 3, 15
            else:
                min_score, max_score = 15, 30

            if score <= min_score:
                return 0
            elif score >= max_score:
                return 1
            else:
                return (score - min_score) / (max_score - min_score)

        fkgl = fkgl_score(policy_gen, self.cmu_dict)
        fkgl = normal_fkgl(fkgl, type)

        # sari, *_ = SARIsent(input_sent, policy_sent, [tgt_sent])
        reward = 0
        if type == 'simp':
            meteor = single_meteor_score(reference=input_seq, hypothesis=policy_gen)
            reward = self.params.delta * meteor + (1 - self.params.delta) * (1 - fkgl)
            # reward = 0.3 * meteor - 0.3 * fkgl * 0.4 * sari
        elif type == 'comp':
            meteor = single_meteor_score(reference=policy_gen, hypothesis=input_seq)
            reward = self.params.delta * meteor + (1 - self.params.delta) * fkgl
            # reward = 0.3 * meteor + 0.3 * fkgl * 0.4 * sari

        return reward

    def get_sari_back(self, ori, pred, ref, ids):
        pred_list = pred.squeeze(0).tolist()
        ref_list = ref.squeeze(0).tolist()
        ori_list = ori.squeeze(0).tolist()

        pred_seq = [self.index2word[num] for num in pred_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]
        ref_seq = [self.index2word[num] for num in ref_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]
        ori_seq = [self.index2word[num] for num in ori_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]

        ori_sent = self.merge_subword(ori_seq)
        pred_sent = self.merge_subword(pred_seq)
        ref_sent = self.merge_subword(ref_seq)
        self.ref_list.append([ref_sent.split()])
        self.output_list.append(pred_sent.split())

        if ids == 0:
            print("ORI: ", ori_sent)
            print("PRE: ", pred_sent)
            print("REF: ", ref_sent)

        return SARIsent(ori_sent, pred_sent, [ref_sent])

    def get_sari(self, ori, pred, ref, ids):
        pred_list = pred.squeeze(0).tolist()
        ref_list = ref.squeeze(0).tolist()
        ori_list = ori.squeeze(0).tolist()

        pred_seq = [self.index2word[num] for num in pred_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]
        ref_seq = [self.index2word[num] for num in ref_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]
        ori_seq = [self.index2word[num] for num in ori_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]

        ori_sent = self.merge_subword(ori_seq)
        pred_sent = self.merge_subword(pred_seq)
        ref_sent = self.merge_subword(ref_seq)
        self.ref_list.append([ref_sent.split()])
        self.output_list.append(pred_sent.split())

        # reward = self.get_reward(ori_sent, pred_sent, ref_sent)

        keep, dels, add = SARIsent(ori_sent, pred_sent, [ref_sent])
        return keep, dels, add

    def lm_eval(self, type):
        if self.lm is None:
            return

        self.lm.eval()
        total_loss = 0
        params = self.params

        with torch.no_grad():
            loader = self.get_loader('lm', type, None)

            for step, batch in enumerate(loader):
                input_seq, input_pos = map(lambda x: x.to(Constants.device), batch)
                tgt_seq = input_seq[:, 1:].contiguous()
                input_seq = input_seq[:, :-1].contiguous()
                loss = self.lm(input_seq, tgt_seq)

                total_loss += loss

            total_loss = total_loss / step

        logger.info("lm_loss: " + str(total_loss.item()))
        return total_loss.item()

    def auto_encoder_eval(self, type):
        if self.model is None:
            return

        self.model.eval()
        total_loss = 0
        params = self.params

        with torch.no_grad():
            loader = self.get_loader('encdec', type, None)

            for step, batch in enumerate(loader):
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(Constants.device), batch)

                batch_size = src_seq.size(0)
                seq_len = tgt_seq.size(1)

                _, seq_logits = self.model.generate(
                    src_seq=src_seq,
                    src_pos=src_pos,
                    src_id=self.type_dict[type],
                    tgt_id=self.type_dict[type],
                    max_len=params.len_max_seq,
                    mode='auto_encoder',
                )
                seq_logits = seq_logits[:, :seq_len-1].contiguous().view(-1, seq_logits.size(-1))
                gold = tgt_seq[:, 1:].contiguous().view(-1)
                loss = F.cross_entropy(seq_logits, gold, ignore_index=Constants.PAD, reduction='sum') / batch_size

                total_loss += loss.item()

        logger.info("auto_encoder loss: " + str(total_loss / step))

    def enc_dec_eval(self, src_type, tgt_type, use_pointer=False, mode='dev'):
        if self.model is None:
            return

        self.model.eval()
        params = self.params
        with torch.no_grad():
            loader = self.get_loader('encdec', src_type, tgt_type, mode=mode)

            t_keep, t_del, t_add = np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 3))
            # t_sari, t_keep, t_del, t_add = [], [], [], []
            for step, batch in enumerate(loader):
                
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(Constants.device), batch)

                pred, _ = self.model.generate(
                    src_seq=src_seq,
                    src_pos=src_pos,
                    src_id=self.type_dict[src_type],
                    tgt_id=self.type_dict[tgt_type],
                    max_len=params.len_max_seq,
                    mode='translate',
                    device=Constants.device,
                )
                keep, dels, add = self.get_sari(src_seq[:, 1:-1], pred, tgt_seq[:, 1:-1], step)
                t_keep = t_keep + keep
                t_del = t_del + dels
                t_add = t_add + add
                #t_sari.append(sent_sari)
                #t_keep.append(keep)
                #t_del.append(dels)
                #t_add.append(add)

            score, keep, dels, add = calculate_suff(t_keep, t_del, t_add)
            #score = mean(t_sari)
            #keep = mean(t_keep)
            #dels = mean(t_del)
            #add = mean(t_add)
            bleu = corpus_bleu(self.ref_list, self.output_list)

        logger.info('average_sari: ' + str(score))
        logger.info('average_bleu: ' + str(bleu))

        '''
        path = os.path.join(self.params.dump_path, self.params.name + 'eval_reward.log')
        with codecs.open(path, mode='a') as f:
            f.write(str(rewards) + '\n')
        '''

        path = os.path.join(self.params.dump_path, self.params.name + 'sari.log')
        with codecs.open(path, mode='a') as f:
            f.write(" ".join([str(score), str(keep), str(dels), str(add)]) + '\n')
        return float(score), float(bleu)

    def eval_all(self, use_pointer, mode='dev'):
        # self.auto_encoder_eval('comp')
        # self.auto_encoder_eval('simp')
        scores = {}
        sari, bleu = self.enc_dec_eval('comp', 'simp', use_pointer=use_pointer, mode=mode)
        scores['sari'] = sari
        scores['bleu'] = bleu
        self.ref_list.clear()
        self.output_list.clear()
        return scores
