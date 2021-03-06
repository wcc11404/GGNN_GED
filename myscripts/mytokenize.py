from stanfordcorenlp import StanfordCoreNLP
import argparse
import tqdm

# 对原始训练语料进行tokenize
def tokenize_(args):
    nlp = StanfordCoreNLP(args.stanford, memory='4g')
    # in_path = ['../data/orign_data/fce-public.train.original.tsv', '../data/orign_data/fce-public.dev.original.tsv',
    #            '../data/orign_data/fce-public.test.original.tsv']
    # out_path = ['../data/process/fce-public.train.preprocess.tsv', '../data/process/fce-public.dev.preprocess.tsv',
    #             '../data/process/fce-public.test.preprocess.tsv']

    special_pattern = {"gonna": ["gon", "na"], "wanna": ["wan", "na"]}
    # for ip, op in zip(in_path, out_path):
    #   with open(ip, 'r') as f1, open(op, 'w') as f2:
    #       lines = f1.readlines()
    #       for num, line in enumerate(lines):
    f1 = open(args.input,'r')
    f2 = open(args.output, 'w')
    lines = f1.readlines()
    for num,line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            f2.write("\n")
            continue
        line = line.split("\t")
        if line[0].lower() in special_pattern and len(lines[num+1].strip()) != 0:
            # 斯坦福那个tokenize竟然是按照上下文切词的，导致有一些特例
            for word in special_pattern[line[0].lower()]:
                f2.write(word + "\t" + line[1] + "\n")
        else:
            wordlist = nlp.word_tokenize(line[0])
            for word in wordlist:
                f2.write(word+"\t"+line[1]+"\n")
    f1.close()
    f2.close()
    nlp.close()

# 特例，去除括号内容
def remove_brackets(line):
    while("(" in line):
        left = line.find("(")
        right = line.find(")")
        if right != -1:
            if right > left:
                if left - 1 >= 0 and right + 1 < len(line):
                    if line[left - 1] == " " and line[right + 1] == " ":
                        left -= 1
                line = line[:left] + line[right + 1:]
            else:
                if left - 1 >= 0 and line[left - 1] == " ":
                    left -= 1
                line = line[:left]
        else:
            if left - 1 >= 0 and line[left - 1] == " ":
                left -= 1
            line = line[:left]
    return line

# 对单语语料进行tokenize
def tokenize__(args):
    f1 = open(args.input, 'r', encoding='utf-8')    # 包含非ascii码的需要这样打开文件
    f2 = open(args.output, 'w')
    lines = f1.read().strip().split("\n")
    for line in tqdm.tqdm(lines):
        line = line.strip()
        if len(line) == 0:  #空行
            continue

        # 检查是否有非ascii码字符
        line = line.encode("utf-8") # str -> bytes
        try:
            line.decode("ascii")
        except UnicodeDecodeError:
            continue
        line = bytes.decode(line)   # bytes -> str

        # 去除括号内容
        line = remove_brackets(line)

        # 特例去除
        if "http" in line or "www" in line:
            continue

        line = line.split()
        if len(line) > 60 or len(line) <= 8:  # 长度不符合
            continue

        # 特例，结尾重复标点
        temp = -1
        for i in range(len(line) - 2, -1, -1):
            if line[i]==line[i+1]:
                temp = i
            else:
                break
        if temp != -1:
            line = line[:temp+1]

        f2.write(" ".join(line))
        f2.write("\n")
    f1.close()
    f2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/orign_data/tete")
    parser.add_argument("--output", default="../data/process/te")
    parser.add_argument("--stanford", default="../data/stanford-corenlp-full-2018-10-05")
    parser.add_argument("--mode", type=int, default=1) # 0==BIO 1==parallel

    args = parser.parse_args()
    if args.mode == 0:
        tokenize_(args)
    elif args.mode == 1:
        tokenize__(args)