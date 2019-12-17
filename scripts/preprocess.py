from stanfordcorenlp import StanfordCoreNLP
import os
import argparse
from collections import Counter
import pickle

#   生成data/train_graph.txt
def parse(deplist):
    skip_relation_list = ["ROOT"]
    graph = {}
    max_index = 0
    for dep in deplist:
        relation, src, tgt = dep
        max_index = max(max_index, src, tgt)
        if relation in skip_relation_list:
            continue
        instr = str(tgt) + "in"
        outstr = str(src) + "out"
        if instr in graph:
            graph[instr] = graph[instr][:] + [str(src) + "," + relation]
        else:
            graph[instr] = [str(src) + "," + relation]
            graph[str(tgt) + "out"] = []
        if outstr in graph:
            graph[outstr] = graph[outstr][:] + [str(tgt) + "," + relation]
        else:
            graph[outstr] = [str(tgt) + "," + relation]
            graph[str(src) + "in"] = []

    # graph的形状 dict["idin"]= ["sid,relation",...]
    # 表明第id个单词有一条从sid指过来的边，边的关系为relation
    return graph, max_index #这个maxindex不对

def graph_to_file(graph_maxindex, file_path):
    with open(file_path, 'w') as f:
        for graph, max_index in graph_maxindex:
            for i in range(1, max_index + 1):
                s = str(i) + "in"
                temp = str(i) + ";"
                if s in graph:
                    temp += " ".join(graph[s])
                temp += ";"
                s = str(i) + "out"
                if s in graph:
                    temp += " ".join(graph[s])
                f.write(temp + "\n")
            f.write("\n")

def generate_graph(mode=0):
    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    nlp = StanfordCoreNLP(r'../data/stanford-corenlp-full-2018-10-05', memory='4g')
    if mode == 0:
        in_path = ['../data/process/fce-public.train.preprocess.tsv', '../data/process/fce-public.dev.preprocess.tsv',
                   '../data/process/fce-public.test.preprocess.tsv']
    elif mode == 1:
        in_path = ['../data/orign_data/fce-public.train.original.tsv', '../data/orign_data/fce-public.dev.original.tsv',
                   '../data/orign_data/fce-public.test.original.tsv']
    out_path = ['../data/process/train_graph.txt', '../data/process/dev_graph.txt', '../data/process/test_graph.txt']

    for ip, op in zip(in_path, out_path):
        f = open(ip, 'r').read().strip().split('\n\n')
        graph_maxindex = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split("\n")
            temp = []
            for wordtuple in line:
                wordtuple = wordtuple.split("\t")
                temp.append(wordtuple[0])
            dp = nlp.dependency_parse(" ".join(temp))
            g, _ = parse(dp)

            ### 加入语序边,每个单词（除首尾）都有一个出边和一个入边
            # id 从1-len(temp)
            outstr = str(1) + "out"
            if outstr in g:
                g[outstr] = g[outstr][:] + [str(2) + ",nextout"]
            else:
                g[outstr] = [str(2) + ",nextout"]

            for num in range(2, len(temp)):
                instr = str(num) + "in"
                if instr in g:
                    g[instr] = g[instr][:] + [str(num - 1) + ",nextin"]
                else:
                    g[instr] = [str(num - 1) + ",nextin"]
                outstr = str(num) + "out"
                if outstr in g:
                    g[outstr] = g[outstr][:] + [str(num + 1) + ",nextout"]
                else:
                    g[outstr] = [str(num + 1) + ",nextout"]

            instr = str(len(temp)) + "in"
            if instr in g:
                g[instr] = g[instr][:] + [str(len(temp) - 1) + ",nextin"]
            else:
                g[instr] = [str(len(temp) - 1) + ",nextin"]

            graph_maxindex.append((g, len(temp)))
        graph_to_file(graph_maxindex, op)
    nlp.close()
#############

# 对原始训练语料进行tokenize
def tokenize_():
    nlp = StanfordCoreNLP(r'../data/stanford-corenlp-full-2018-10-05', memory='4g')
    in_path = ['../data/orign_data/fce-public.train.original.tsv', '../data/orign_data/fce-public.dev.original.tsv',
               '../data/orign_data/fce-public.test.original.tsv']
    out_path = ['../data/process/fce-public.train.preprocess.tsv', '../data/process/fce-public.dev.preprocess.tsv',
                '../data/process/fce-public.test.preprocess.tsv']
    for ip, op in zip(in_path, out_path):
        with open(ip, 'r') as f1, open(op, 'w') as f2:
            for line in f1:
                line = line.strip()
                if len(line) == 0:
                    f2.write("\n")
                    continue
                line = line.split("\t")
                wordlist = nlp.word_tokenize(line[0])
                for word in wordlist:
                    f2.write(word+"\t"+line[1]+"\n")
    nlp.close()

# 读取训练数据
def load(fpath, use_lower=False):
    # if not os.path.exists(fpath):
    #     raise FileNotFoundError("Can not find the file \""+fpath+"\"")
    templist = open(fpath).read().strip().split("\n\n")
    x = []
    y = []
    size = []
    for sentence in templist:
        wordlist = []
        labellist = []
        sentence = sentence.split("\n")
        for wordpair in sentence:
            wordpair = wordpair.split("\t")
            if use_lower:
                wordpair[0] = wordpair[0].lower()
            wordlist.append(wordpair[0])
            labellist.append(wordpair[1])
        x.append(wordlist)
        y.append(labellist)
        size.append(len(wordlist))
    return x, y, size  # ['Dear', 'Sir', 'or', 'Madam', ',']  ['c', 'c', 'c', 'c', 'c'] 5

# 读取图数据
def load_graph(fpath):
    f = open(fpath, 'r').read().strip().split("\n\n")
    graph_in = []
    graph_out = []
    for line in f:
        line = line.split("\n")
        graph_sen_in = []
        graph_sen_out = []
        for word in line:
            word = word.split(";")
            edge_in = word[1].split()
            edge_in = [[int(edge.split(',')[0]), edge.split(',')[1]] for edge in edge_in]
            edge_out = word[2].split()
            edge_out = [[int(edge.split(',')[0]), edge.split(',')[1]] for edge in edge_out]
            graph_sen_in.append(edge_in)
            graph_sen_out.append(edge_out)
        graph_in.append(graph_sen_in)
        graph_out.append(graph_sen_out)
    return [graph_in, graph_out] # in/out , sen , word , [id->relation]

# 生成词表
def makeword2veclist(datasetlist):
    counter = Counter()
    for dataset in datasetlist:
        for instance in dataset:
            counter.update(instance)
    word2id = {}
    id2word = []
    word2id["<pad>"] = 0
    word2id["<unk>"] = 1
    id2word.append("<pad>")
    id2word.append("<unk>")
    num = len(id2word)
    for k, v in counter.most_common():
        word2id[k] = num
        id2word.append(k)
        num += 1
    return word2id, id2word

# 生成字符表
def makechar2veclist(datasetlist):
    counter = Counter()
    for dataset in datasetlist:
        for instance in dataset:
            for word in instance:
                counter.update(word)
    char2id = {}
    id2char = []
    char2id["<pad>"] = 0
    char2id["<unk>"] = 1
    id2char.append("<pad>")
    id2char.append("<unk>")
    num = len(id2char)
    for k, v in counter.most_common():
        char2id[k] = num
        id2char.append(k)
        num += 1
    return char2id, id2char

# 生成边表
def makeedge2veclist(datasetlist):
    counter = Counter()
    for dataset in datasetlist:
        #其实只用统计入度边即可
        for instance in dataset[0]:
                for word in instance:
                    for id_relation in word:
                        relation = id_relation[1]
                        counter.update([relation])
        for instance in dataset[1]:
                for word in instance:
                    for id_relation in word:
                        relation = id_relation[1]
                        counter.update([relation])
    edge2id = {}
    id2edge = []
    # edge2id["<pad>"] = 0 # 这个pad已经没用了
    edge2id["<unk>"] = 0#1
    # id2edge.append("<pad>")
    id2edge.append("<unk>")
    num = len(id2edge)
    for k, v in counter.most_common():
        edge2id[k] = num
        id2edge.append(k)
        num += 1
    return edge2id, id2edge

def lookup_word(dataset, word2id, label2id, ispad=False, padid=0, unkid=1, padlabel=-1, maxlength=150):
    x = []
    for instance in dataset[0]:
        temp = [word2id[w] if w in word2id else unkid for w in instance]
        if ispad:
            temp = temp[:min(len(temp), maxlength)]
            temp.extend([padid for _ in range(maxlength - len(temp))])
        x.append(temp)

    y = []
    for instance in dataset[1]:
        temp = [label2id[w] for w in instance]
        if ispad:
            temp = temp[:min(len(temp), maxlength)]
            temp.extend([padlabel for _ in range(maxlength - len(temp))])
        y.append(temp)

    return x, y

def lookup_char(dataset, char2id, ispad=False, padid=0, unkid=1, maxcharlength=-1):
    charx = []
    charsize = []
    for instance in dataset:
        ctemp = []
        sizetemp = []
        for w in instance:
            cc = [char2id[c] if c in char2id else unkid for c in w]
            ctemp.append(cc)
            sizetemp.append(len(cc))
        charx.append(ctemp)
        charsize.append(sizetemp)
    return charx, charsize

def lookup_graph(dataset, edge2id, ispad=False, padid=0, unkid=1):
    graph = []
    for in_out in dataset:
        graph_in_out = []
        for instance in in_out:
            graph_sen = []
            for word in instance:
                graph_word = []
                for id_relation in word:
                    if id_relation[1] in edge2id:
                        graph_word.append([id_relation[0], edge2id[id_relation[1]]])
                    else:
                        graph_word.append([id_relation[0], unkid])
                graph_sen.append(graph_word)
            graph_in_out.append(graph_sen)
        graph.append(graph_in_out)
    return graph

def main(args):
    def save_preprocess(dir):
        f = open(dir, 'wb')

        pickle.dump(word2id, f)
        pickle.dump(id2word, f)
        pickle.dump(char2id, f)
        pickle.dump(id2char, f)
        pickle.dump(edge2id, f)
        pickle.dump(id2edge, f)

        pickle.dump(trainx, f)
        pickle.dump(trainy, f)
        pickle.dump(trainsize, f)
        pickle.dump(trainx_char, f)
        pickle.dump(trainsize_char, f)
        pickle.dump(train_graph, f)

        pickle.dump(devx, f)
        pickle.dump(devy, f)
        pickle.dump(devsize, f)
        pickle.dump(devx_char, f)
        pickle.dump(devsize_char, f)
        pickle.dump(dev_graph, f)

        pickle.dump(testx, f)
        pickle.dump(testy, f)
        pickle.dump(testsize, f)
        pickle.dump(testx_char, f)
        pickle.dump(testsize_char, f)
        pickle.dump(test_graph, f)

        f.close()

    # 预处理的预处理
    # tokenize_()
    # generate_graph(mode=0)

    # 读取原始数据
    trainx, trainy, trainsize = load(args.data_dir + r"/process/fce-public.train.preprocess.tsv", bool(args.use_lower))
    devx, devy, devsize = load(args.data_dir + r"/process/fce-public.dev.preprocess.tsv", bool(args.use_lower))
    testx, testy, testsize = load(args.data_dir + r"/process/fce-public.test.preprocess.tsv", bool(args.use_lower))

    train_graph = load_graph(args.data_dir + r"/process/train_graph.txt")
    dev_graph = load_graph(args.data_dir + r"/process/dev_graph.txt")
    test_graph = load_graph(args.data_dir + r"/process/test_graph.txt")

    # 统计各种表
    word2id, id2word = makeword2veclist([trainx, devx])
    char2id, id2char = makechar2veclist([trainx, devx])
    edge2id, id2edge = makeedge2veclist([train_graph, dev_graph])
    label2id = {"c": 0, "i": 1}

    # 查表替换
    trainx_char, trainsize_char = lookup_char(trainx, char2id, ispad=False) # 一定要先处理
    trainx, trainy = lookup_word((trainx, trainy), word2id, label2id, ispad=False)
    train_graph = lookup_graph(train_graph, edge2id)

    devx_char, devsize_char = lookup_char(devx, char2id, ispad=False)
    devx, devy = lookup_word((devx, devy), word2id,label2id, ispad=False)
    dev_graph = lookup_graph(dev_graph, edge2id)

    testx_char, testsize_char = lookup_char(testx, char2id, ispad=False)
    testx, testy = lookup_word((testx, testy), word2id,label2id, ispad=False)
    test_graph = lookup_graph(test_graph, edge2id)

    # 序列化
    save_preprocess(args.preprocess_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--use-lower", default="True")
    parser.add_argument("--preprocess-dir", default="data/preprocess.pkl")
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    main(args)