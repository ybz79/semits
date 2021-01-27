import torch
import numpy as np
import torch.utils.data as data
from model import Constants
from dataset.tokenization import BertTokenizer
import re
import codecs


def paired_collate_fn(insts):
    src_insts, tgt_insts = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    tgt_insts = collate_fn(tgt_insts)
    return (*src_insts, *tgt_insts)


def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    batch_pos = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    batch_seq = torch.Tensor(batch_seq).long()
    batch_pos = torch.Tensor(batch_pos).long()

    return batch_seq, batch_pos


class MonoLingualData(data.Dataset):
    def __init__(self, params, mono_data, word2index, max_len,
                 frequent_word_list=None, ppdb_rules=None, data_mode='simp', train_mode='autoencoder'):
        self.mono = mono_data
        self.word2index = word2index
        self.max_len = max_len
        self.tokenizer = BertTokenizer(params.vocab_path)
        self.params = params
        self.frequent_word_list = frequent_word_list
        self.ppdb_rules = ppdb_rules
        self.data_mode = data_mode
        self.word_ids = None
        self.train_mode = train_mode
        self.stop_set, self.light_set = self.load_stoplist()

    def load_stoplist(self):
        path = self.params.stoplist_path
        f = codecs.open(path).readlines()
        k = f + ['.', ',', ';', '\'', '`', '*', '?', '\\', '\\\\']
        stop_set = set(k)
        light_set = set(f)
        return stop_set, light_set
        

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return Constants.UNK

    def add_noise(self, seq, word_ids):
        if self.data_mode =='simp':
            if self.params.additive:
                seq, word_ids = self.word_additive(seq, word_ids)
                seq = self.word_shuffle(seq, word_ids, word_ids[-1], flag=1)
            else:
                seq = self.word_shuffle(seq, word_ids, self.params.word_shuffle)
                seq = self.word_drop(seq)
        else:
            seq = self.word_shuffle(seq, word_ids, self.params.word_shuffle)
            seq = self.word_drop(seq)

        return " ".join(seq)

    def word_shuffle(self, seq, word_ids, degree, flag=0):
        noise = np.random.uniform(0, degree, len(seq))
        select_noise = noise[word_ids]
        if flag:
            score = select_noise + 1e-6 * np.arange(len(seq))
        else:
            score = select_noise + word_ids + 1e-6 * np.arange(len(seq))

        index = score.argsort()
        ret_seq = [seq[i] for i in index]
        return ret_seq

    def word_replace(self, seq, rules):
        for key, values in rules.items():
            if key in seq and np.random.rand() <= self.params.word_replace and key not in self.stop_set:
                index = np.random.randint(0, len(values))
                alternative_words = values[index]
                if alternative_words in ['.', ',', ';', '\'', '`', '*', '?', '\\', '\\\\']:
                    continue
                try:
                    seq = re.sub(key, alternative_words, seq, count=1)
                except:
                    # print("error occured, the key is ", key, alternative_words)
                    pass
        
        return seq.split()

    def word_additive(self, seq, word_ids):
        index = np.random.randint(0, self.__len__())
        additive_seq = self.mono[index].split()
        if self.params.shuffle_mode == 'unigram':
            additive_word_ids = np.arange(len(additive_seq), dtype=int)
        elif self.params.shuffle_mode == 'bigram':
            additive_word_ids = np.array([int(num / 2) for num in range(len(additive_seq))])
        else:
            additive_word_ids = self.word_ids[index]

        min = int((additive_word_ids[-1] + 1) * 0.3)
        max = int((additive_word_ids[-1] + 1) * 0.6)

        additive_len = np.random.randint(min, max+1)
        sampled_ids = np.random.choice(additive_word_ids[-1]+1, additive_len, replace=False)

        sampled_seq = [additive_seq[i] for i in range(len(additive_seq)) if additive_word_ids[i] in sampled_ids]
        ad_word_ids = [additive_word_ids[i] for i in range(len(additive_seq))
                       if additive_word_ids[i] in sampled_ids]
        index = 0
        if len(ad_word_ids) > 0:
            pre = ad_word_ids[0]
        for i in range(len(ad_word_ids)):
            if ad_word_ids[i] != pre:
                index += 1
            pre = ad_word_ids[i]
            ad_word_ids[i] = index + word_ids[-1] + 1

        return seq + sampled_seq, np.concatenate((word_ids, np.array(ad_word_ids, dtype=int)), axis=0)

    def word_drop(self, seq):
        keep = np.random.rand(len(seq)) >= self.params.word_dropout
        frequent_mask = [w not in self.frequent_word_list for w in seq]

        if self.params.drop_type == 'Zero':
            new_seq = [w for j, w in enumerate(seq) if keep[j]]
        else:
            new_seq = [w for j, w in enumerate(seq) if (keep[j] or frequent_mask[j])]
        if len(new_seq) == 0:
            new_seq.insert(1, seq[np.random.randint(1, len(seq)-1)])
        return new_seq

    def __getitem__(self, item):
        mono_seq = self.mono[item]
        corupt_seq = mono_seq

        mono_seq = self.tokenizer.tokenize(mono_seq)

        mono_seq = [self.get_index(word) for word in mono_seq]
        if len(mono_seq) > self.max_len - 2:
            mono_seq = mono_seq[:self.max_len - 2]
        
        if self.data_mode == 'simp':
            mono_seq = [Constants.SBOS] + mono_seq + [Constants.EOS]
        else:
            mono_seq = [Constants.CBOS] + mono_seq + [Constants.EOS]

        if self.train_mode == 'autoencoder':
            rules = self.ppdb_rules[item]
            corupt_seq = self.word_replace(corupt_seq, rules)

            if self.params.shuffle_mode == 'unigram':
                word_ids = np.arange(len(corupt_seq), dtype=int)
            elif self.params.shuffle_mode == 'bigram':
                word_ids = np.array([int(num/2) for num in range(len(corupt_seq))])
            else:
                word_ids = self.word_ids[item]
            assert len(corupt_seq) == len(word_ids)

            corupt_seq = self.add_noise(seq=corupt_seq, word_ids=word_ids)
            corupt_seq = self.tokenizer.tokenize(corupt_seq)

            corupt_seq = [self.get_index(word) for word in corupt_seq]
            if len(corupt_seq) > self.max_len - 2:
                corupt_seq = corupt_seq[:self.max_len - 2]

            if self.data_mode == 'simp':
                corupt_seq = [Constants.SBOS] + corupt_seq + [Constants.EOS]
            else:
                corupt_seq = [Constants.CBOS] + corupt_seq + [Constants.EOS]

            return corupt_seq, mono_seq
        
        return mono_seq

    def __len__(self):
        return len(self.mono)


class ParallelData(data.Dataset):
    def __init__(self, complex_sent, simp_sent, word2index, max_len, mode='train'):
        self.comp_sent = complex_sent
        self.simp_sent = simp_sent

        self.word2index = word2index
        self.mode = mode
        self.max_len = max_len

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return Constants.UNK

    def __getitem__(self, item):
        com_seq = self.comp_sent[item]
        sim_seq = self.simp_sent[item]

        com_seq = [self.get_index(word) for word in com_seq]
        if len(com_seq) > self.max_len - 2:
            com_seq = com_seq[:self.max_len - 2]
        sim_seq = [self.get_index(word) for word in sim_seq]
        if len(sim_seq) > self.max_len - 2:
            sim_seq = com_seq[:self.max_len - 2]

        com_seq = [Constants.CBOS] + com_seq + [Constants.EOS]
        sim_seq = [Constants.SBOS] + sim_seq + [Constants.EOS]

        return com_seq, sim_seq

        
    def __len__(self):
        return len(self.comp_sent)

