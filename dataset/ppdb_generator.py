import codecs
import re
import pickle
import numpy as np
import time
import math


def read_ppdb(path, pp_path):
    comp2simp_dict = {}
    simp2comp_dict = {}

    ppdb_file = codecs.open(path, mode='r')
    for step, i in enumerate(ppdb_file.readlines()):
        rules = i.strip().split('\t')
        score = float(rules[1])
        comp_word = re.sub("be", "", rules[3])
        simp_word = re.sub("be", "", rules[4])

        if score > 0.7:
            if comp_word in comp2simp_dict:
                if simp_word not in comp2simp_dict[comp_word] and len(comp2simp_dict[comp_word]) < 5:
                    comp2simp_dict[comp_word].append(simp_word)
            else:
                comp2simp_dict[comp_word] = [simp_word]

            if simp_word in simp2comp_dict:
                if comp_word not in simp2comp_dict[simp_word] and len(simp2comp_dict[simp_word]) < 5:
                    simp2comp_dict[simp_word].append(comp_word)
            else:
                simp2comp_dict[simp_word] = [comp_word]

        if step % 100000 == 0:
            print("handel" , step , "lines")

    print("adding simpleppdb")

    ppdbpp_file = codecs.open(pp_path, mode='r')
    for step, i in enumerate(ppdbpp_file.readlines()):
        rules = i.strip().split('\t')
        score = float(rules[2])
        
        if math.fabs(score) > 0.6:
            if score > 0:
                comp_word = rules[0]
                simp_word = rules[1]
            else:
                comp_word = rules[1]
                simp_word = rules[0]

            if comp_word in comp2simp_dict:
                if simp_word not in comp2simp_dict[comp_word] and len(comp2simp_dict[comp_word]) <= 5:
                    comp2simp_dict[comp_word].append(simp_word)
            else:
                comp2simp_dict[comp_word] = [simp_word]

            if simp_word in simp2comp_dict:
                if comp_word not in simp2comp_dict[simp_word] and len(simp2comp_dict[simp_word]) <= 5:
                    simp2comp_dict[simp_word].append(comp_word)
            else:
                simp2comp_dict[simp_word] = [comp_word]
        if step % 100000 == 0:
            print("handel", step, "lines")

    comp2simp_file = codecs.open("comp2simp.pkl", mode='wb')
    simp2comp_file = codecs.open("simp2comp.pkl", mode='wb')

    pickle.dump(comp2simp_dict, comp2simp_file)
    pickle.dump(simp2comp_dict, simp2comp_file)


def generate_rules(path, dic_path):
    origin_file = codecs.open(path)
    rule_dict = codecs.open(dic_path, mode='rb')
    rule_dict = pickle.load(rule_dict)
    grams = [3, 2, 1]
    rules = []

    a = time.time()
    for step, i in enumerate(origin_file):
        line = i.strip().split()
        keep = np.ones(len(line))

        sentence_dict = {}
        for gram in grams:
            for index in range(len(line) - gram + 1):
                if keep[index:index+gram].sum() == gram:
                    query = " ".join(line[index:index+gram])
                    if query in rule_dict:
                        sentence_dict[query] = rule_dict[query]
                        keep[index:index+gram] = 0

        rules.append(sentence_dict)
        if step == 0:
            print(sentence_dict)
        if step % 100000 == 0:
            print(step, "lines")

    f = codecs.open("comp_rules.pkl", mode='wb')
    pickle.dump(rules, f)


if __name__ == '__main__':
    generate_rules('../data/mono/new_comp.txt', 'comp2simp.pkl')
    # read_ppdb("../data/SimplePPDB", "../data/SimplePPDBpp")

