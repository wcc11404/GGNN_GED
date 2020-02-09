#python scripts/mytokenize.py --mode 1 --input data/orign_data/1b.train.txt --output data/process/pretrain.train --stanford data/stanford-corenlp-full-2018-10-05
#python scripts/mytokenize.py --mode 1 --input data/orign_data/1b.dev.txt --output data/process/pretrain.dev --stanford data/stanford-corenlp-full-2018-10-05

head -n 1000000 data/process/pretrain.train > data/process/pretrain.train.1M
python scripts/genGraph.py --mode 1 --input data/process/pretrain.train.1M --output data/process/pretrain.train.graph --process data/process.pretrain.train.ic --stanford data/stanford-corenlp-full-2018-10-05
python scripts/genGraph.py --mode 1 --input data/process/pretrain.dev --output data/process/pretrain.dev.graph --process data/process.pretrain.dev.ic --stanford data/stanford-corenlp-full-2018-10-05

# 预处理的预处理
# tokenize_()
# generate_graph(mode=0)

#binary

#pretrian

#train

#test
