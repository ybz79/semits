import numpy as np
import sys
import codecs
from metrics.STAR import SARIsent
import re


def normalize(line):
    line = re.sub("&quot", "\"", line)
    line = re.sub("&amp", "&", line)
    line = re.sub("&lt", "<", line)
    line = re.sub("&gt", ">", line)
    line = re.sub("&apos", "'", line)

    split_on = ["!", "\"", "#", "\$", "%", "&", "\(", "\)", "\*", "\+", "/", ":", ";", "<", "=", ">", "\?", "@", "\[", "\]", "\^", "_", "`", "\{", "\|", "\}", "~", "\\\\"]
    i = 0
    while(i < len(split_on)):
        regex = split_on[i]
        line = re.sub(regex, " " + regex + " ", line)
        i += 1

    line = " " + line + " "
    line = re.sub("\\s+", " ", line)

    split_index = set()
    for i in range(len(line)):
        ch = line[i]
        if ch == '.' or ch == ',':
            prev_ch = line[i-1]
            next_ch = line[i+1]
            if prev_ch < '0' or prev_ch > '9' or next_ch < '0' or next_ch > '9':
                split_index.add(i)
        elif ch == '-':
            prev_ch = line[i-1]
            if prev_ch >= '0' and prev_ch <= '9':
                split_index.add(i)

    line0 = line
    line = ""

    for i in range(len(line0)):
        if i in split_index:
            line += " " + line0[i] + " "
        else:
            line += line0[i]

    line = " " + line + " "
    line = re.sub("\\s+", " ", line)

    line = line.strip()

    return line

    

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
    # print(score / 3)
    return sum(keep_f) / 4, sum(del_f) / 4, sum(add_f) / 4, score / 3


def get_sari(comp_path, simp_path, ref_path, num_ref, verbose=False):
    comp_file = codecs.open(comp_path).readlines()
    simp_file = codecs.open(simp_path).readlines()
    if num_ref > 1:
        ref_list = []
        for i in range(num_ref):
            ref_file = codecs.open(ref_path + '.' + str(i)).readlines()
            ref_list.append(ref_file)

    else:
        ref_list = [codecs.open(ref_path).readlines()]

    t_keep, t_del, t_add = np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 3))
    index = 0.
    for i in range(len(comp_file)):
        comp_sent = comp_file[i].strip().lower()
        simp_sent = normalize(simp_file[i].strip().lower())
        ret_sent_list = [normalize(ref_list[j][i].strip().lower()) for j in range(num_ref)]

        keep, dels, add = SARIsent(comp_sent, simp_sent, ret_sent_list)
        t_keep = t_keep + keep
        t_del = t_del + dels
        t_add = t_add + add
        
        if verbose:
            calculate_suff(keep, dels, add)
            
    return calculate_suff(t_keep, t_del, t_add)


if __name__ == '__main__':
    # output_path = './aaai_result/wiki/Dress-Ls.lower'
    # ref_path = './turkcorpus/test.8turkers.tok.turk'
    # src_path = './turkcorpus/test.8turkers.tok.norm'
    # num_refs = 8

    output_path = sys.argv[1]
    ref_path = sys.argv[2]
    src_path = sys.argv[3]
    num_refs = int(sys.argv[4])

    sari = get_sari(src_path, output_path, ref_path, num_refs, True)
    print(sari)
