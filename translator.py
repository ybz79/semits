import torch
from torch.nn import functional as F
import codecs
import os
import model.Constants as Constants
from metrics.SARI import SARIsent
from logging import getLogger
from nltk.translate.bleu_score import corpus_bleu

logger = getLogger()


class Evaluator(object):

    def __init__(self, model, lm, data, params):
        self.model = model
        self.lm = lm
        self.data = data
        self.index2word = data['index2word']
        self.params = params
        self.type_dict = {'comp': 0, 'simp': 1}
        self.index2word = data['index2word']
        self.ref_list = []
        self.output_list = []

    def get_loader(self, loader_name, src_type, tgt_type, mode='dev'):
        if loader_name == 'encdec':
            if tgt_type is None:
                data_loader = self.data['mono'][loader_name][src_type + '_dev']
            else:
                data_loader = self.data['para'][mode]
        else:
            data_loader = self.data['lm'][src_type + '_dev']
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

    def get_sari(self, ori, pred, ref, ids):
        pred_list = pred.squeeze(0).tolist()
        ref_list = ref.squeeze(0).tolist()
        ori_list = ori.squeeze(0).tolist()

        pred_seq = [self.index2word[num] for num in pred_list if num not in [Constants.PAD, Constants.EOS, Constants.SBOS, Constants.CBOS]]
        ref_seq = [self.index2word[num] for num in ref_list if num not in [Constants.PAD, Constants.EOS, Constants.CBOS, Constants.SBOS]]
        ori_seq = [self.index2word[num] for num in ori_list if num not in [Constants.PAD, Constants.EOS, Constants.CBOS, Constants.SBOS]]

        ori_sent = self.merge_subword(ori_seq)
        pred_sent = self.merge_subword(pred_seq)
        ref_sent = self.merge_subword(ref_seq)
        self.ref_list.append([ref_sent.split()])
        self.output_list.append(pred_sent.split())

        a = codecs.open(self.params.output_name, mode='a')
        a.write(pred_sent.strip() + '\n')

        return SARIsent(ori_sent, pred_sent, [ref_sent])

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
        total_sari = 0
        total_keep = 0
        total_del = 0
        total_add = 0
        with torch.no_grad():
            loader = self.get_loader('encdec', src_type, tgt_type, mode=mode)
            for step, batch in enumerate(loader):
                
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(Constants.device), batch)
                if params.beam_size > 0:
                    generate_func = self.model.generate_beam_search
                else:
                    generate_func = self.model.generate

                pred, _ = generate_func(
                    src_seq=src_seq,
                    src_pos=src_pos,
                    src_id=self.type_dict[src_type],
                    tgt_id=self.type_dict[tgt_type],
                    max_len=params.len_max_seq,
                    mode='translate',
                    device=Constants.device,
                )
                sari, avgkeepscore, avgdelscore, avgaddscore = self.get_sari(src_seq[:, 1:-1], pred, tgt_seq[:, 1:-1], step)
                total_sari += sari
                total_keep += avgkeepscore
                total_del += avgdelscore
                total_add += avgaddscore

            bleu = corpus_bleu(self.ref_list, self.output_list)
            print(bleu)

        # logger.info('average_sari: ' +  str(total_sari / step))
        # logger.info('average_keep: ' +  str(total_keep / step))
        # logger.info('average_del: ' +  str(total_del / step))
        # logger.info('average_add: ' +  str(total_add / step))
        
       # path = os.path.join(self.params.dump_path, self.params.name + 'sari.log')
       # with codecs.open(path, mode='a') as f:
       #     f.write(" ".join([str(total_sari / step), str(total_keep / step), str(total_del / step), str(total_add / step)]) + '\n')
       # return total_sari / step

    def eval_all(self, use_pointer, mode='dev'):
        # self.auto_encoder_eval('comp')
        # self.auto_encoder_eval('simp')
        scores = {}
        sari = self.enc_dec_eval('comp', 'simp', use_pointer=use_pointer, mode=mode)
        scores['sari'] = sari
        return scores
