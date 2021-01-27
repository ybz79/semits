import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model.Constants as Constants
from model.Modules import MultiHeadAttention, PositionwiseFeedForward
import pickle
import codecs
from logging import getLogger

logger = getLogger()


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.n_src_vocab = args.n_src_vocab

        n_enc_layers = args.n_enc_layers
        d_word_vec = args.emb_dim

        n_position = 300

        self.embedding = nn.Embedding(
            self.n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        if args.us_pretrain_embedding:
            self.init_embedding(args)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList()
        for k in range(n_enc_layers):
            layer_is_shared = (k >= (args.n_enc_layers - args.share_enc))
            if layer_is_shared:
                logger.info("Sharing encoder transformer parameters for layer %i" % k)

            self.layer_stack.append(nn.ModuleList([
                # layer for first complex sentence
                EncoderLayer(args=args)
            ]))
            if layer_is_shared:
                self.layer_stack[k].append(self.layer_stack[k][0])
            else:
                self.layer_stack[k].append(EncoderLayer(args=args))

    def init_embedding(self, args):
        logger.info("init embeddings with pretrained vector")
        pre_train_path = args.embedding_path
        weight_file = codecs.open(pre_train_path, mode='rb')
        emb_weight = pickle.load(weight_file)

        self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=True)

    def forward(self, src_seq, src_pos, input_id, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.embedding(src_seq) + self.position_enc(src_pos)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer[input_id](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()

        self.dropout = args.dropout
        self.n_tgt_vocab = args.n_tgt_vocab
        self.share_encdec_emb = args.share_encdec_emb
        self.share_dec = args.share_dec
        self.freeze_dec_emb = args.freeze_dec_emb

        self.encoder_class = encoder.__class__
        self.beam_size = args.beam_size

        n_dec_layers = args.n_dec_layers
        d_word_vec = args.emb_dim
        n_position = 300

        self.embedding = nn.Embedding(
            self.n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        if args.us_pretrain_embedding:
            self.init_embedding(args)        

        if self.share_encdec_emb:
            logger.info("Sharing encoder and decoder input embeddings")
            self.embedding.weight = encoder.embedding.weight

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList()

        for k in range(n_dec_layers):
            layer_is_shared = (k < self.share_dec)
            if layer_is_shared:
                logger.info("Sharing decoder transformer parameters for layer %i" % k)

            self.layer_stack.append(nn.ModuleList([
                DecoderLayer(args=args)
            ]))
            if layer_is_shared:
                self.layer_stack[k].append(self.layer_stack[k][0])
            else:
                self.layer_stack[k].append(DecoderLayer(args=args))

    def init_embedding(self, args):
        logger.info("init embeddings with pretrained vector")
        pre_train_path = args.embedding_path
        weight_file = codecs.open(pre_train_path, mode='rb')
        emb_weight = pickle.load(weight_file)

        self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=False)

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, input_id, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.embedding(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer[input_id](
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_enc_attn_list, dec_slf_attn_list
        return dec_output,


class EncoderLayer(nn.Module):
    # Compose with two layers

    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.d_model = args.d_model
        self.d_inner = args.d_inner
        self.n_head = args.n_head
        self.d_k = args.d_k
        self.d_v = args.d_v
        self.dropout = args.dropout

        self.slf_attn = MultiHeadAttention(
            self.n_head, self.d_model, self.d_k, self.d_v, dropout=self.dropout)
        self.pos_ffn = PositionwiseFeedForward(self.d_model, self.d_inner, dropout=self.dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.d_model = args.d_model
        self.d_inner = args.d_inner
        self.n_head = args.n_head
        self.d_k = args.d_k
        self.d_v = args.d_v

        self.slf_attn = MultiHeadAttention(self.n_head, self.d_model, self.d_k, self.d_v, dropout=args.dropout)
        self.enc_attn = MultiHeadAttention(self.n_head, self.d_model, self.d_k, self.d_v, dropout=args.dropout)
        self.pos_ffn = PositionwiseFeedForward(self.d_model, self.d_inner, dropout=args.dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    # Sinusoid position encoding table

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.Tensor(sinusoid_table).float()


def get_attn_key_pad_mask(seq_k, seq_q):
    # For masking out the padding part of key sequence.

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    # For masking out the subsequent info.

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
