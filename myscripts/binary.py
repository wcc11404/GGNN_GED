import argparse
import pickle
import random

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
        m = [[trainx, trainy, trainsize], [trainx_char, trainsize_char], [train_graph],
             [devx, devy, devsize], [devx_char, devsize_char], [dev_graph],
             [testx, testy, testsize], [testx_char, testsize_char], [test_graph]]

        assert sign[0]==1 and sign[3]==1
        item0 = sign[0] + sign[3] + sign[6]
        item1 = sign[1] + sign[4] + sign[7]
        item2 = sign[2] + sign[5] + sign[8]
        assert item0 == item1 and item1 == item2

        f = open(dir, 'wb')
        pickle.dump(sign, f)
        for i, j in enumerate(sign):
            if j == 1:
                for item in m[i]:
                    pickle.dump(item, f)

        f.close()

    if len(args.train_graph_dir) !=0 and len(args.train_dir) != len(args.train_graph_dir):
        raise ValueError()

    # 读取原始数据
    trainx, trainy, trainsize = [], [], []
    for dir in args.train_dir:
        x, y, s = load(dir, args.use_lower)
        trainx.extend(x)
        trainy.extend(y)
        trainsize.extend(s)
    random.seed(44)
    random.shuffle(trainx)
    random.seed(44)
    random.shuffle(trainy)
    random.seed(44)
    random.shuffle(trainsize)
    devx, devy, devsize = load(args.dev_dir, args.use_lower)
    if args.test_dir is not None:
        testx, testy, testsize = load(args.test_dir, args.use_lower)
    else:
        testx, testy, testsize = None, None, None

    trainx_char, trainsize_char = None, None
    devx_char, devsize_char = None, None
    testx_char, testsize_char = None, None
    train_graph, dev_graph, test_graph = None, None, None
    if args.edge_vocab_dir is not None:
        if args.train_graph_dir is not None:
            train_graph_in, train_graph_out = [], []
            for dir in args.train_graph_dir:
                re = load_graph(dir)
                train_graph_in.extend(re[0])
                train_graph_out.extend(re[1])
            random.seed(44)
            random.shuffle(train_graph_in)
            random.seed(44)
            random.shuffle(train_graph_out)
            train_graph = [train_graph_in, train_graph_out]

        if args.dev_graph_dir is not None:
            dev_graph = load_graph(args.dev_graph_dir)
        if args.test_graph_dir is not None:
            test_graph = load_graph(args.test_graph_dir)

    f = open(args.word_vocab_dir, "rb")
    word2id = pickle.load(f)
    id2word = pickle.load(f)
    f.close()
    if args.char_vocab_dir is not None:
        f = open(args.char_vocab_dir, "rb")
        char2id = pickle.load(f)
        id2char = pickle.load(f)
        f.close()
    if args.edge_vocab_dir is not None:
        f = open(args.edge_vocab_dir, "rb")
        edge2id = pickle.load(f)
        id2edge = pickle.load(f)
        f.close()
    label2id = {"c": 0, "i": 1}

    # 查表替换
    sign = [1, 0, 0, 1, 0, 0, 0, 0, 0]
    if args.char_vocab_dir is not None:
        trainx_char, trainsize_char = lookup_char(trainx, char2id, ispad=False) # 一定要先处理
        sign[1] = 1
    trainx, trainy = lookup_word((trainx, trainy), word2id, label2id, ispad=False)
    if args.edge_vocab_dir is not None and args.train_graph_dir is not None:
        train_graph = lookup_graph(train_graph, edge2id)
        sign[2] = 1
    print("train data finish")

    if args.char_vocab_dir is not None:
        devx_char, devsize_char = lookup_char(devx, char2id, ispad=False)
        sign[4] = 1
    devx, devy = lookup_word((devx, devy), word2id, label2id, ispad=False)
    if args.edge_vocab_dir is not None and args.dev_graph_dir is not None:
        dev_graph = lookup_graph(dev_graph, edge2id)
        sign[5] = 1
    print("dev data finish")

    if args.test_dir is not None:
        if args.char_vocab_dir is not None:
            testx_char, testsize_char = lookup_char(testx, char2id, ispad=False)
            sign[7] = 1
        testx, testy = lookup_word((testx, testy), word2id, label2id, ispad=False)
        sign[6] = 1
        if args.edge_vocab_dir is not None and args.test_graph_dir is not None:
            test_graph = lookup_graph(test_graph, edge2id)
            sign[8] = 1
        print("test data finish")

    # 序列化
    save_preprocess(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", nargs="+", required=True)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--test-dir")

    parser.add_argument("--train-graph-dir", nargs="+")
    parser.add_argument("--dev-graph-dir", default="../data/process/dev_graph.txt")
    parser.add_argument("--test-graph-dir")

    parser.add_argument("--word-vocab-dir", default="../data/preprocess.pkl", required=True)
    parser.add_argument("--char-vocab-dir", default="../data/preprocess.pkl")
    parser.add_argument("--edge-vocab-dir", default="../data/preprocess.pkl")

    parser.add_argument("--use-lower", action='store_true', default=True)
    parser.add_argument("--output", default="../data/preprocess.pkl", required=True)
    args = parser.parse_args()
    main(args)