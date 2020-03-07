import argparse
import tqdm
from collections import Counter
import random
import xml.dom.minidom
import os

def corrupt_sentence(args):
    f = open(args.input,"r").read().split("\n\n")
    counter = Counter()
    for line in f:
        line = line.split("\n")
        for word in line:
            word = word.split("\t")
            counter.update([word[0]])

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

def readXMLfile(filepath):
    dom = xml.dom.minidom.parse(filepath)
    root = dom.documentElement
    dd = root.getElementsByTagName("NS")

    skip = [".", ",", "?", "!", "-", "\'", "\"", ":", ";", " "]
    re = []
    for item in dd:
        incor = item.firstChild
        cor = item.lastChild
        if incor is None or cor is None:
            continue
        if incor.nodeName == "i" and cor.nodeName == "c":
            try:
                incordata = incor.childNodes[0]._data.strip().lower()
                cordata = cor.childNodes[0]._data.strip().lower()
                if len(incordata) == 1 and incordata in skip:
                    continue
                if len(cordata) == 1 and cordata in skip:
                    continue
                if incordata == cordata:
                    continue
                if len(cordata.split()) > 4:
                    continue
                re.append([incordata, cordata])
            except:
                pass
    return re

def loaddict(dir):
    result = {}
    for root, dirs, files in os.walk(dir):
        for file in files:
            dir = os.path.join(root, file)
            re = readXMLfile(dir)
            for item in re:
                if item[1] in result:
                    if item[0] not in result[item[1]]:
                        temp = result[item[1]][:]
                        temp.append(item[0])
                        result[item[1]] = temp
                else:
                    result[item[1]] = [item[0]]
    return result

def findlist(l1, l2, num=0):
    if num >= len(l1):
        return -1
    i = num-1
    while (i < len(l1)):
        i += 1
        try:
            i = l1.index(l2[0], i, len(l1) - 1)
        except ValueError:
            return -1
        flag = True
        for j in range(1, len(l2)):
            if l2[j] != l1[i + j]:
                flag = False
                break
        if flag:
            return i
    return -1

def replacelist(l1, l2, x, num):
    minnum = min(len(l2), num)
    for i in range(minnum):
        l1[x + i] = l2[i]
    if len(l2) > minnum:
        for i in range(len(l2) - minnum):
            l1.insert(x + minnum + i, l2[minnum + i])
    if num > minnum:
        for i in range(num - minnum):
            del l1[x + minnum]
    return l1

def pattern_corrupt(args):
    dic = loaddict(args.dict_dir)
    f = open(args.input, "r").read().strip().split("\n")
    f1 = open(args.output, "w")
    tj = 0
    sum = 0
    i=0
    for line in tqdm.tqdm(f):
        i+=1
        if i>100000:
            break
        line = line.strip().lower()
        line = line.split()
        label = ["c" for _ in range(len(line))]
        line = " ".join(line)
        for k, v in dic.items():
            x = 0
            while (k in line[x:]):
                r = random.random()
                if r > 1:  # 选择是否腐化
                    break
                r = random.randint(0, len(v) - 1)  # 用哪个腐化
                temp = v[r].split()  # 这个是要替换的
                line = line.split()
                kk = k.split()
                x = findlist(line, kk, x)   # 从x的位置开始查找list line中是否有list kk
                if x == -1: # 没找到退出
                    line = " ".join(line)
                    break
                line = replacelist(line, temp, x, len(kk))
                label = replacelist(label, ["i" for _ in range(len(temp))], x, len(kk))
                line = " ".join(line)
                x += len(temp)  # x位置偏移
                tj += len(temp)

        line = line.split()
        sum += len(line)
        assert len(line) == len(label)
        for word, lab in zip(line, label):
            f1.write(word + "\t" + lab + "\n")
        f1.write("\n")
    f1.close()
    print(tj)
    print(sum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/temp.txt")
    parser.add_argument("--output", default="../data/shit.txt")
    parser.add_argument("--errorrate", type=float, default=0.2)
    parser.add_argument("--dict-dir", default="../data/orign_data/fce-released-dataset/dataset")
    parser.add_argument("--mode", type=int, default=1) # 0==corrupt 1==pattern

    args = parser.parse_args()
    if args.mode == 0:
        corrupt_sentence(args)
    elif args.mode == 1:
        pattern_corrupt(args)