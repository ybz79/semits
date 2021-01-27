from curses.ascii import isdigit
from nltk.tokenize import RegexpTokenizer
from string import digits
import re


def remove_digits(sent):
    trans = str.maketrans('', '', digits)
    return sent.translate(trans)


def word_count(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    word_seq = tokenizer.tokenize(sent)
    return word_seq, len(word_seq)


def get_syllable(word, d):
    if word in d:
        return [len(list(y for y in x if isdigit(y[-1]))) for x in d[word.lower()]][0]
    else:
        vowels = "aeiouy"
        numVowels = 0
        lastWasVowel = False
        for wc in word:
            foundVowel = False
            for v in vowels:
                if v == wc:
                    if not lastWasVowel:
                        numVowels += 1
                    foundVowel = lastWasVowel = True
                    break
            if not foundVowel:
                lastWasVowel = False
        if len(word) > 2 and word[-2:] == "es":
            numVowels -= 1
        elif len(word) > 1 and word[-1:] == "e":
            numVowels -= 1
        return numVowels


def fkgl_score(sentence, cmu_dict):
    sentence = re.sub("[\(\[{]+[^\(\)\[\]{}]*[\)\]}]+", "", sentence)
    if len(sentence) == 0:
        return 0

    pure_sent = remove_digits(sentence)

    word_seq, word_num = word_count(pure_sent)
    if word_num == 0:
        return 0
    sentence_num = 1
    syllable_count = 0

    for word in word_seq:
        temp_count = get_syllable(word, cmu_dict)
        syllable_count += temp_count

    score = 0.39 * (word_num / sentence_num) + 11.8 * (syllable_count / word_num) - 15.59
    return score
