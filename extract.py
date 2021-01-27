import codecs
import sys

def extract(path, output_path):
    w_ref = codecs.open(output_path + '/ref.txt', mode='w')
    w_comp = codecs.open(output_path + '/comp.txt', mode='w')
    with codecs.open(path) as f:
        refs = f.readlines()
        for l in refs:
            line = l.strip().split('|')
            w_comp.write(line[0].strip() + '\n')
            w_ref.write(line[1].strip() + '\n')


if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
