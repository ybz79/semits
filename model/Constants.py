import numpy as np
import torch

PAD = 0
UNK = 100
SBOS = 101
CBOS = 101
EOS = 102
LABEL_PAD = 3

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
CBOS_WORD = '[CCLS]'
SBOS_WORD = '[SCLS]'
EOS_WORD = '[SEP]'

vocab_path = './data/google/voca.list'
pos_path = './pos.list'

train_path = './data/google/formal.train'
valid_path = './data/google/formal.dev'
test_path = './data/google/formal.dev'

trained_path= './trained_model'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

max_oov = 15
Pointer=True
