from stanfordcorenlp import StanfordCoreNLP
import argparse
import tqdm
from collections import Counter
import random

def corrupt_sentence(args):
    f = open(args.input,"r").read().split("\n\n")
    counter = Counter()
    for line in f:
        line = line.split("\n")
        for word in line:
            word = word.split("\t")
            counter.update(word[0])

    voc = []
    for k, v in counter.most_common():
        voc.append(k)

    with open(args.output, "w") as f1:
        for line in tqdm.tqdm(f):
            line = line.split("\n")
            re = []
            for word in line:
                r = random.random()
                if r <= args.errorrate:
                    w = random.randint(0, len(voc) - 1)
                    re.append(voc[w] + "\t" + "i")
                else:
                    re.append(word)
            f1.write("\n".join(re))
            f1.write("\n\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/temp.txt")
    parser.add_argument("--output", default="../data/shit.txt")
    parser.add_argument("--errorrate", type=float, default=0.2)
    #parser.add_argument("--stanford", default="../data/stanford-corenlp-full-2018-10-05")
    #parser.add_argument("--mode", type=int, default=1) # 0==BIO 1==parallel

    args = parser.parse_args()
    corrupt_sentence(args)