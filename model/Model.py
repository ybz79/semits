import torch
import torch.nn as nn
import torch.nn.functional as F
import model.Constants as Constants
from torch.distributions import Categorical
from model.Transformer import Encoder, Decoder
import logging
from model.Beam import Beam
import numpy as np

logger = logging.getLogger()


def otf_batch(batch_hyp):
    batch_size = len(batch_hyp)
    index = np.random.choice(4, batch_size, replace=True, p=[0.4, 0.3, 0.2, 0.1])
    chosen_batch = [batch_hyp[i][index[i]] for i in range(batch_size)]

    max_len = len(max(chosen_batch, key=lambda x:len(x)))
    batch_seq = np.array([
        seq + [Constants.PAD] * (max_len - len(seq))
        for seq in chosen_batch])

    batch_pos = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(seq)] for seq in batch_seq])

    batch_seq = torch.Tensor(batch_seq).long()
    batch_pos = torch.Tensor(batch_pos).long()

    return batch_seq, batch_pos


class Transformer(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.args = args
        self.n_head = args.n_head
        self.d_model = args.d_model
        self.n_src_vocab = args.n_src_vocab
        self.n_tgt_vocab = args.n_tgt_vocab

        self.share_decpro_emb = args.share_decpro_emb
        self.share_output_emb = args.share_output_emb

        self.encoder = Encoder(args=args)
        self.decoder = Decoder(args=args, encoder=self.encoder)

        self.tgt_copy_rate = nn.Linear(self.d_model * 3, 1, bias=True)
        self.attn_weight = nn.Parameter(torch.ones(self.n_head))

        proj = [nn.Linear(self.d_model, self.n_tgt_vocab, bias=True) for _ in range(2)]
        for i in proj:
            nn.init.xavier_normal_(i.weight)

        if self.share_decpro_emb:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            for i in range(2):
                proj[i].weight = self.decoder.embedding.weight
            self.x_logit_scale = (self.d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if self.share_output_emb:
            logger.info("Sharing decoder projection matrices")
            proj[1].weight = proj[0].weight
            proj[1].bias = proj[0].bias

        self.tgt_word_prj = nn.ModuleList(proj)

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, src_id, tgt_id):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos, src_id)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, tgt_id)
        seq_logit = self.tgt_word_prj[tgt_id](dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))

    def generate(self, src_seq, src_pos, src_id, tgt_id, max_len, mode='otf', device=Constants.device):
        with torch.no_grad():
            enc_output, *_ = self.encoder(src_seq, src_pos, src_id)
            batch_size = src_seq.size(0)
            
            if tgt_id == 0:
                BOS = Constants.CBOS
            else:
                BOS = Constants.SBOS
            tgt = torch.ones(batch_size, 1).fill_(BOS).long().to(device)
            tgt_pos = torch.ones(1).long().expand(batch_size, -1).to(device)
            eos_mask = torch.ones(batch_size, 1).byte().to(device)

            seq_logit = torch.Tensor().to(device)
            for i in range(max_len-1):
                dec_output, *_ = self.decoder(tgt, tgt_pos, src_seq, enc_output, tgt_id)
                prob = self.tgt_word_prj[tgt_id](dec_output[:, -1]) * self.x_logit_scale
                
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.unsqueeze(-1).long().masked_fill(1 - eos_mask, Constants.PAD).to(device)

                tgt = torch.cat((tgt, next_word), dim=-1)
                tgt_pos = torch.arange(1, i + 3).expand(batch_size, -1).to(device)
                eos_mask *= (next_word != Constants.EOS)

                if mode != 'otf':
                    seq_logit = torch.cat((seq_logit, prob.unsqueeze(1)), dim=1)
                else:
                    if torch.sum(eos_mask) == 0:
                        break

        if mode == 'otf':
            return tgt, tgt_pos
        else:
            return tgt, seq_logit

    def policy_generate(self, src_seq, src_pos, src_id, tgt_id, max_len, device=Constants.device):
        enc_output, *_ = self.encoder(src_seq, src_pos, src_id)
        batch_size = src_seq.size(0)

        tgt = torch.ones(batch_size, 1).fill_(Constants.BOS).long().to(device)
        tgt_pos = torch.ones(1).long().expand(batch_size, -1).to(device)
        eos_mask = torch.ones(batch_size, 1).byte().to(device)

        log_probs = []
        entropy = 0
        for i in range(max_len-1):
            dec_output, *_ = self.decoder(tgt, tgt_pos, src_seq, enc_output, tgt_id)
            prob = self.tgt_word_prj[tgt_id](dec_output[:, -1]) * self.x_logit_scale
            prob = F.softmax(prob, dim=-1)

            distribution = Categorical(prob)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            log_probs.append(log_prob.unsqueeze(-1))

            action = action.unsqueeze(-1).long().masked_fill(1 - eos_mask, Constants.PAD).to(device)
            tgt = torch.cat((tgt, action), dim=-1)

            tgt_pos = torch.arange(1, i + 3).expand(batch_size, -1).to(device)
            eos_mask *= (action != Constants.EOS)
            entropy += distribution.entropy().mean()

            if torch.sum(eos_mask) == 0:
                break

        return tgt[:, 1:], log_probs, entropy

    def generate_beam_search(self, src_seq, src_pos, src_id, tgt_id, max_len, mode='otf', device=Constants.device):
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):

                dec_output, *_ = self.decoder(dec_seq, dec_pos, src_seq, enc_output, tgt_id)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(self.tgt_word_prj[tgt_id](dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            n_bm = self.args.beam_size
            enc_output, *_ = self.encoder(src_seq, src_pos, src_id)

            batch_size, seq_len, d_h = enc_output.size()
            src_seq = src_seq.repeat(1, n_bm).view(batch_size*n_bm, seq_len)
            enc_output = enc_output.repeat(1, n_bm, 1).view(batch_size*n_bm, seq_len, d_h)

            dec_beams = [Beam(n_bm, type=tgt_id, device=device) for _ in range(batch_size)]
            active_index_list = list(range(batch_size))
            index2position_map = get_inst_idx_to_tensor_position_map(active_index_list)

            for i in range(max_len):
                active_index_list = beam_decode_step(
                    dec_beams, i+1, src_seq, enc_output, index2position_map, n_bm
                )

                if not active_index_list:
                    break
                src_seq, enc_output, index2position_map = collate_active_info(
                    src_seq, enc_output, index2position_map, active_index_list
                )

        if mode == 'otf':
            batch_hyp, batch_scores = collect_hypothesis_and_scores(dec_beams, 4)
            return otf_batch(batch_hyp)
        else:
            batch_hyp, batch_scores = collect_hypothesis_and_scores(dec_beams, 1)
            return torch.Tensor(batch_hyp[0]).long(), 0
