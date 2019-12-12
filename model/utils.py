import torch
import os

def savecheckpoint(model, dir):
    torch.save(model.state_dict(), dir)

def loadcheckpoint(model, dir):
    if not os.path.exists(dir):
        raise KeyError("checkpoint is not exist")
    model.load_state_dict(torch.load(dir))

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

def generate_graph():
    from stanfordcorenlp import StanfordCoreNLP
    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05', memory='4g')
    file_path = '../data/fce-public.train.original.tsv'
    f = open(file_path, 'r').read().split('\n\n')
    graph_maxindex = []
    for line in f:
        line = line.split("\n")
        temp = []
        for wordtuple in line:
            wordtuple = wordtuple.split("\t")
            temp.append(wordtuple[0])
        dp = nlp.dependency_parse(" ".join(temp))
        g, m = parse(dp)
        graph_maxindex.append((g, len(temp)))
    graph_to_file(graph_maxindex, "temp.txt")
    nlp.close()
#############
