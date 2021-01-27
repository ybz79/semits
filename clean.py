import codecs
import random
import re
import os

def select_by_index(path, index):
    a = codecs.open(path).readlines()
    b = codecs.open(path + 'sample', mode='w')
    for i in index:
        b.write(a[i].strip() + '\n')
    return

def double(matched):
    value = matched.group('value')
    return re.sub(" , ", "," ,value)

def double2(matched):
    value = matched.group('value')
    return re.sub(" . ", ".", value)

def clean(sent):
    sent = re.sub("\' \'", "\'\'", sent)
    sent = re.sub("` `", "\'\'", sent)
    sent = re.sub(" - - ", " -- ", sent)
    sent = re.sub(" \' s ", " \'s ", sent)
    sent = re.sub(" \' ve ", " \'ve ", sent)
    sent = re.sub(" \' d ", " \'d ", sent)
    sent = re.sub(" \' ll ", " \'ll ", sent)
    sent = re.sub(" - ", "-", sent)
    sent = re.sub(" n \' t ", " n\'t ", sent)
    sent = re.sub(" \' m ", " \'m ", sent)
    sent = re.sub(" u . s . ", " u.s. ", sent)
    sent = re.sub(" h . w . ", " h.w. ", sent)
    sent = re.sub(" st . ", " st. ", sent)
    sent = re.sub(" mrs . ", " mrs. ", sent)
    sent = re.sub("< sep >", "<sep>", sent)
    sent = re.sub('(?P<value>(\d+ , \d+))', double, sent)
    sent = re.sub('(?P<value>(\d+ \. \d+))', double2, sent)

    return sent

        

def random_chose(paths):
    '''
    index = []
    for i in range(1000):
        if random.random() < 0.1:
            index.append(i)
    '''
    for path in paths:
       # select_by_index(path, index)
        if os.path.exists(path):
            b = codecs.open(path).readlines()
            a = codecs.open(path + 'smooth', mode='w')
            for i in b:
                sent = clean(i.strip())
                a.write(sent + '\n')

if __name__ == '__main__':
    dataset = './final_result/wiki/untswikismooth'
    paths  = [dataset]
    random_chose(paths)
        
    
