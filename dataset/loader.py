from torch.utils.data import DataLoader
from dataset.dataset import MonoLingualData, ParallelData, collate_fn, paired_collate_fn
from dataset.tokenization import BertTokenizer
import codecs
import os
from logging import getLogger
import pickle

logger = getLogger()


def load_mono_data(params, vocab):
    mono_data_list = ['simp_train', 'simp_dev', 'comp_train', 'comp_dev']
    mono_data_path = []
    mono_data = {'encdec':{}, 'otf':{}}
    mono_data_path.append(params.simp_train_path)
    mono_data_path.append(params.simp_dev_path)
    mono_data_path.append(params.comp_train_path)
    mono_data_path.append(params.comp_dev_path)

    for path, name in zip(mono_data_path, mono_data_list):
        assert os.path.isfile(path), path
        logger.info("Loading data from %s ..." % path)
        with codecs.open(path) as f:
            read_file = f.readlines()
            raw_data = [sent.strip() for sent in read_file]
            data_mode = None

            if 'comp' in name:
                logger.info("Loading data from %s ..." % params.comp_frequent_list)
                frequent_list = load_frequent_list(params.comp_frequent_list)
                logger.info("Loading data from %s ..." % params.comp_ppdb_rules)
                ppdb_rules = load_ppdb_rules(params.comp_ppdb_rules)
                data_mode = 'comp'

            elif 'simp' in name:
                frequent_list = load_frequent_list(params.simp_frequent_list)
                logger.info("Loading data from %s ..." % params.simp_ppdb_rules)
                ppdb_rules = load_ppdb_rules(params.simp_ppdb_rules)
                data_mode = 'simp'

            else:
                frequent_list, ppdb_rules = None, None

            loader = DataLoader(
                dataset=MonoLingualData(
                    params=params,
                    mono_data=raw_data,
                    word2index=vocab,
                    max_len=params.len_max_seq,
                    frequent_word_list=frequent_list,
                    ppdb_rules=ppdb_rules,
                    data_mode=data_mode,
                    train_mode='autoencoder'
                ),
                batch_size=params.batch_size,
                shuffle=True,
                collate_fn=paired_collate_fn,
            )

            otf_loader = DataLoader(
                dataset=MonoLingualData(
                    params=params,
                    mono_data=raw_data,
                    word2index=vocab,
                    max_len=params.len_max_seq,
                    frequent_word_list=frequent_list,
                    ppdb_rules=ppdb_rules,
                    data_mode=data_mode,
                    train_mode='otf'
                ),
                batch_size=params.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )
            mono_data['encdec'][name] = loader
            mono_data['otf'][name] = otf_loader
            
    return mono_data


def load_frequent_list(path):
    frequent_word_list = set()
    with codecs.open(path) as f:
        for w in f.readlines():
            if w not in frequent_word_list:
                frequent_word_list.add(w)

    return frequent_word_list


def load_ppdb_rules(path):
    rule_files = codecs.open(path, mode='rb')
    ppdb_rules = pickle.load(rule_files)

    return ppdb_rules


def load_parallel_data(params, vocab):
    para_data_list = ['dev', 'test']
    para_data_path = []
    
    para_data = {}
    para_data_path.append(params.para_dev_path)
    para_data_path.append(params.para_test_path)

    if params.supervised_rate > 0:
        para_data_list.append('train')
        para_data_path.append(params.para_train_path)

    tokenizer = BertTokenizer(params.vocab_path)
    
    for path, name in zip(para_data_path, para_data_list):
        assert os.path.isfile(path)
        logger.info("Loading data from %s ..." % path)
        with codecs.open(path) as f:
            read_filev = f.readlines()
            comp_sents = []
            simp_sents = []
            for i in read_filev:
                line = i.strip().split('|')
                comp_sents.append(tokenizer.tokenize(line[0]))
                simp_sents.append(tokenizer.tokenize(line[1]))

            if name == 'train':
                batch_size = params.batch_size
            else:
                batch_size = 1

            loader = DataLoader(
                dataset=ParallelData(simp_sent=simp_sents, complex_sent=comp_sents, word2index=vocab, max_len=params.len_max_seq),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=paired_collate_fn,
            )
            para_data[name] = loader

    return para_data


def load_vocab(params):
    vocab_path = params.vocab_path
    logger.info(vocab_path)
    assert os.path.isfile(vocab_path)

    vocab = []
    with codecs.open(vocab_path) as f:
        read_file = f.readlines()
        for i in read_file:
            vocab.append(i.strip())

    return dict(zip(vocab, range(len(vocab)))), dict(zip(range(len(vocab)), vocab))


def load_data(params):
    data = dict()
    word2index, index2word = load_vocab(params)
    data['index2word'] = index2word
    data['word2index'] = word2index
    data['mono'] = load_mono_data(params, word2index)
    data['para'] = load_parallel_data(params, word2index)

    return data

