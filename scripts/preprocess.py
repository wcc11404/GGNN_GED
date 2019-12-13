from stanfordcorenlp import StanfordCoreNLP

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
            g, m = parse(dp)
            graph_maxindex.append((g, len(temp)))
        graph_to_file(graph_maxindex, op)
    nlp.close()
#############

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

if __name__ == "__main__":
    tokenize_()
    generate_graph(mode=0)