#python scripts/mytokenize.py --mode 1 --input data/orign_data/1b.train.txt --output data/process/pretrain.train --stanford data/stanford-corenlp-full-2018-10-05
#python scripts/mytokenize.py --mode 1 --input data/orign_data/1b.dev.txt --output data/process/pretrain.dev --stanford data/stanford-corenlp-full-2018-10-05

#head -n 1000000 data/process/pretrain.train > data/process/pretrain.train.1M
#head -n 20000 data/process/pretrain.dev > data/process/pretrain.dev.20K
#python scripts/genGraph.py --mode 1 --input data/process/pretrain.train.1M --output data/process/pretrain.train.1M.graph --process data/process/pretrain.train.1M.ic --stanford data/stanford-corenlp-full-2018-10-05
#python scripts/genGraph.py --mode 1 --input data/process/pretrain.dev.20K --output data/process/pretrain.dev.20K.graph --process data/process/pretrain.dev.20K.ic --stanford data/stanford-corenlp-full-2018-10-05

python scripts/genVocab.py --mode 0 --input data/process/fce-public.train.preprocess.tsv data/process/fce-public.dev.preprocess.tsv --output data/prepare/wordvocab.pkl --mergeinput data/process/pretrain.train.1M.ic --mergemaxnum 50000
python scripts/genVocab.py --mode 1 --input data/process/pretrain.train.1M.ic data/process/fce-public.train.preprocess.tsv data/process/fce-public.dev.preprocess.tsv --output data/prepare/charvocab.pkl
python scripts/genVocab.py --mode 2 --input data/process/pretrain.train.1M.graph data/process/train_graph.txt data/process/dev_graph.txt --output data/prepare/edgevocab.pkl
# 预处理的预处理
# tokenize_()
# generate_graph(mode=0)

#binary

#pretrian

#train

#test
