import argparse
import tqdm
from collections import Counter
import pickle

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

def mergewordvocab(addtional_data, maxnum, w2i, i2w):
    counter = Counter()
    for instance in addtional_data:
        counter.update(instance)
    num = len(i2w)
    n = maxnum-num
    for k, v in counter.most_common(n):
        w2i[k] = num
        i2w.append(k)
        num += 1
    return w2i, i2w

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs='+', required=True)
    parser.add_argument("--mergeinput", default=None)
    parser.add_argument("--mergemaxnum", type=int, default=50000)
    parser.add_argument("--use-lower", action='store_true', default=True)
    parser.add_argument("--output", default="../data/preprocess.pkl")
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    out = open(args.output, 'wb')
    if args.mode == 0:
        input = []
        for item in args.input:
            x, _, _ = load(item, args.use_lower)
            input.append(x)
        w2i, i2w = makeword2veclist(input)
        addtional_data, _, _ = load(args.mergeinput, args.use_lower)
        w2i, i2w = mergewordvocab(addtional_data,args.mergemaxnum, w2i,i2w)
        print("word vocabulary: " + str(len(w2i)))
        pickle.dump(w2i, out)
        pickle.dump(i2w, out)
    elif args.mode == 1:
        input = []
        for item in args.input:
            x, _, _ = load(item, args.use_lower)
            input.append(x)
        c2i, i2c = makechar2veclist(input)
        print("char vocabulary: " + str(len(c2i)))
        pickle.dump(c2i, out)
        pickle.dump(i2c, out)
    else:
        input = []
        for item in args.input:
            x = load_graph(item)
            input.append(x)
        e2i, i2e = makeedge2veclist(input)
        print("edge vocabulary: " + str(len(e2i)))
        pickle.dump(e2i, out)
        pickle.dump(i2e, out)
    out.close()