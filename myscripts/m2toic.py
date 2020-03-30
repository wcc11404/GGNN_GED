import argparse

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

def main(args):
    f = open(args.input, "r").read().strip().split("\n\n")
    f1 = open(args.output1, "w")
    f2 = open(args.output2, "w")

    for item in f:
        item = item.strip().split("\n")
        err = item[0].split()[1:]
        label1 = ["c" for _ in range(len(err))]
        label2 = ["c" for _ in range(len(err))]

        for line in item[1:]:
            line = line.split("|||")
            writer_id = int(line[-1])
            start = int(line[0].split()[1])
            end = int(line[0].split()[2])
            if start == -1 and end == -1:
                continue
            elif start == end:
                continue

            if writer_id == 0:
                label1 = replacelist(label1, ["i" for _ in range(end - start)], start, end - start)
            elif writer_id == 1:
                label2 = replacelist(label2, ["i" for _ in range(end - start)], start, end - start)

        assert len(err) == len(label1)
        assert len(err) == len(label2)

        # 处理特例XXXX.XXX这种没切分的情况
        temp = err
        l1 = label1
        l2 = label2
        while(len(temp) != 0):
            flag = False
            for i in range(len(temp)-1):
                if "." in temp[i]:
                    word = temp[i]
                    temp1 = "c"#l1[i]
                    temp2 = "c"#l2[i]
                    err = temp[:i]
                    temp = temp[i + 1:]
                    label1 = l1[:i]
                    label2 = l2[:i]
                    l1 = l1[i + 1:]
                    l2 = l2[i + 1:]
                    if word[0] == ".":
                       temp.insert(0, word[1:])
                       l1.insert(0, temp1)
                       l2.insert(0, temp2)
                    elif word[-1] == ".":
                        err.append(word[:-1])
                        label1.append(temp1)
                        label2.append(temp2)
                    else:
                        p = word.find(".")
                        temp.insert(0, word[p+1:])
                        l1.insert(0, temp1)
                        l2.insert(0, temp2)

                        err.append(word[:p])
                        label1.append(temp1)
                        label2.append(temp2)

                    err.append(".")
                    label1.append(temp1)
                    label2.append(temp2)
                    flag = True
                    break
            if not flag:
                err = temp
                temp = []

            for word, la in zip(err, label1):
                f1.write(word+"\t"+la+"\n")
            f1.write("\n")
            for word, la in zip(err, label2):
                f2.write(word+"\t"+la+"\n")
            f2.write("\n")

    f1.close()
    f2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="E:/conll2014.test.m2")
    parser.add_argument("--output1", default="E:/shit1")
    parser.add_argument("--output2", default="E:/shit2")

    args = parser.parse_args()

    main(args)