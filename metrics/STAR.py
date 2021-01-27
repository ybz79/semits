from __future__ import division
from collections import Counter
import sys
import numpy as np
import re


def ReadInFile (filename):
    
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def sum_counter(counter):
    value = 0.
    for k, v in counter.items():
        value += v
    
    return value


def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)
    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref
        
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref
    
    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keep_scores = np.array([sum_counter(keepgramcountergood_rep), sum_counter(keepgramcounter_rep), sum_counter(keepgramcounterall_rep)])

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    delgramcountergood_rep = delgramcounter_rep & delgramcounterall_rep

    del_scores = np.array([sum_counter(delgramcountergood_rep), sum_counter(delgramcounter_rep), sum_counter(delgramcounterall_rep)])

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    add_scores = np.array([len(addgramcountergood), len(addgramcounter), len(addgramcounterall)])
    return keep_scores, del_scores, add_scores


def SARIsent (ssent, csent, rsents) :
    numref = len(rsents)	

    s1grams = re.split("\\s+", ssent.lower())
    c1grams = re.split("\\s+", csent.lower())
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []
 
    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = re.split("\\s+", rsent.lower())
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams)-1) :
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i+1]
                r2grams.append(r2gram)
            if i < len(r1grams)-2:
                r3gram = r1grams[i] + " " + r1grams[i+1] + " " + r1grams[i+2]
                r3grams.append(r3gram)
            if i < len(r1grams)-3:
                r4gram = r1grams[i] + " " + r1grams[i+1] + " " + r1grams[i+2] + " " + r1grams[i+3]
                r4grams.append(r4gram)        
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)
       
    for i in range(0, len(s1grams)-1) :
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i+1]
            s2grams.append(s2gram)
        if i < len(s1grams)-2:
            s3gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2]
            s3grams.append(s3gram)
        if i < len(s1grams)-3:
            s4gram = s1grams[i] + " " + s1grams[i+1] + " " + s1grams[i+2] + " " + s1grams[i+3]
            s4grams.append(s4gram)
            
    for i in range(0, len(c1grams)-1) :
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i+1]
            c2grams.append(c2gram)
        if i < len(c1grams)-2:
            c3gram = c1grams[i] + " " + c1grams[i+1] + " " + c1grams[i+2]
            c3grams.append(c3gram)
        if i < len(c1grams)-3:
            c4gram = c1grams[i] + " " + c1grams[i+1] + " " + c1grams[i+2] + " " + c1grams[i+3]
            c4grams.append(c4gram)


    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)

    keep_score = np.vstack((keep1score, keep2score, keep3score, keep4score))
    del_score = np.vstack((del1score, del2score, del3score, del4score))
    add_score = np.vstack((add1score, add2score, add3score, add4score))

    return keep_score, del_score, add_score


def main():

    fnamenorm   = "./turkcorpus/test.8turkers.tok.norm"
    fnamesimp   = "./turkcorpus/test.8turkers.tok.simp"
    fnameturk  = "./turkcorpus/test.8turkers.tok.turk."

    ssent = "About 95 species are currently accepted ."
    csent1 = "About 95 you now get in ."
    csent2 = "About 95 species are now agreed ."
    csent3 = "About 95 species are currently agreed ."
    rsents = ["About 95 species are currently known .", "About 95 species are now accepted .", "95 species are now accepted ."]

    print(SARIsent(ssent, csent1, rsents))
    print(SARIsent(ssent, csent2, rsents))
    print(SARIsent(ssent, csent3, rsents))


if __name__ == '__main__':
    main()  

