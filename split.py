import codecs
from dataset.tokenization import BertTokenizer

vocab_path = "data/vocab.list"

def split(path):
    readfile = codecs.open(path)
    writefile = codecs.open(path + 'smooth', mode='w')
    tokenizer = BertTokenizer(vocab_path)

    for i in readfile.readlines():
        line = i.strip()
        sub_list = tokenizer.tokenize(line)
        print(sub_list)

        new_line, _ = merge_subword(sub_list)

        writefile.write(new_line + '\n')


def merge_subword(subword_list):
    ret_sent = []
    align = []
    index = 0
    prev = ""
    for step, word in enumerate(subword_list):
        if "##" in word:
            prev += word.strip("##")
        else:
            if prev != "":
                ret_sent.append(prev)
                index += 1
            prev = word

        align.append(index)

    if prev != "":
        ret_sent.append(prev)

    return " ".join(ret_sent), align


if __name__ == "__main__":
    split("./final_result/wiki/untswiki")
