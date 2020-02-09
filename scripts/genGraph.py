from stanfordcorenlp import StanfordCoreNLP
import argparse
import tqdm

#   解析deplist
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

# 图写入文件
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

# 生成data/train_graph.txt
def generate_graph(args, addcontextedge=False):
    # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
    nlp = StanfordCoreNLP(args.stanford, memory='4g')
    # if mode == 0:
    #     in_path = ['../data/process/fce-public.train.preprocess.tsv', '../data/process/fce-public.dev.preprocess.tsv',
    #                '../data/process/fce-public.test.preprocess.tsv']
    # elif mode == 1:
    #     in_path = ['../data/orign_data/fce-public.train.original.tsv', '../data/orign_data/fce-public.dev.original.tsv',
    #                '../data/orign_data/fce-public.test.original.tsv']
    # out_path = ['../data/process/train_graph.txt', '../data/process/dev_graph.txt', '../data/process/test_graph.txt']
    in_path = args.input
    out_path = args.output

    # for ip, op in zip(in_path, out_path):
    f = open(in_path, 'r').read().strip().split('\n\n')
    graph_maxindex = []
    for line in tqdm.tqdm(f):
        line = line.strip()
        if len(line) == 0:
            break
        line = line.split("\n")
        temp = []
        for wordtuple in line:
            wordtuple = wordtuple.split("\t")
            temp.append(wordtuple[0])
        dp = nlp.dependency_parse(" ".join(temp))
        g, _ = parse(dp)

        ### 加入语序边,每个单词（除首尾）都有一个出边和一个入边
        # id 从1-len(temp)
        # 注test集会有只有一个单词存在的情况
        if addcontextedge == True:
            if len(temp) > 1:
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

            if len(temp) > 1:
                instr = str(len(temp)) + "in"
                if instr in g:
                    g[instr] = g[instr][:] + [str(len(temp) - 1) + ",nextin"]
                else:
                    g[instr] = [str(len(temp) - 1) + ",nextin"]

        graph_maxindex.append((g, len(temp)))
    graph_to_file(graph_maxindex, out_path)
    f.close()
    nlp.close()

# 生成data/pretrain_graph.txt
def generate_graph_(args, addcontextedge=False):
    nlp = StanfordCoreNLP(args.stanford, memory='4g')
    in_path = args.input
    out_path = args.output

    # for ip, op in zip(in_path, out_path):
    f = open(in_path, 'r').read().strip().split('\n')
    process = open(args.process, 'w')
    graph_maxindex = []
    for line in tqdm.tqdm(f):
        line = line.strip()
        if len(line) == 0:
            continue
        # dp = nlp.dependency_parse(" ".join(temp))

        re = nlp._request('depparse', line)['sentences']
        re = re[0]
        sen = [item["originalText"] for item in re["tokens"]]
        dp = [(dep['dep'], dep['governor'], dep['dependent']) for dep in re['basicDependencies']]

        g, _ = parse(dp)

        ### 加入语序边,每个单词（除首尾）都有一个出边和一个入边
        # id 从1-len(temp)
        # 注test集会有只有一个单词存在的情况
        if addcontextedge == True:
            if len(sen) > 1:
                outstr = str(1) + "out"
                if outstr in g:
                    g[outstr] = g[outstr][:] + [str(2) + ",nextout"]
                else:
                    g[outstr] = [str(2) + ",nextout"]

            for num in range(2, len(sen)):
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

            if len(sen) > 1:
                instr = str(len(sen)) + "in"
                if instr in g:
                    g[instr] = g[instr][:] + [str(len(sen) - 1) + ",nextin"]
                else:
                    g[instr] = [str(len(sen) - 1) + ",nextin"]

        # pretrain语料从一行一句话转换为序列标注格式输入文件
        for word in sen:
            process.write(word+"\tc\n")
        process.write("\n")

        graph_maxindex.append((g, len(sen)))
    graph_to_file(graph_maxindex, out_path)
    process.close()
    nlp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data")
    parser.add_argument("--process", default="../data")
    parser.add_argument("--output", default="../data")
    parser.add_argument("--stanford", default="../data/stanford-corenlp-full-2018-10-05")
    parser.add_argument("--addcontextedge", action='store_true', default=False)
    parser.add_argument("--mode", type=int, default=0) # 0==BIO 1==parallel

    args = parser.parse_args()
    if args.mode == 0:
        generate_graph(args, args.addcontextedge)
    elif args.mode == 1:
        generate_graph_(args, args.addcontextedge)