#python scripts/mytokenize.py --mode 1 --input data/orign_data/1b.train.txt --output data/process/pretrain.train --stanford data/stanford-corenlp-full-2018-10-05
#python scripts/mytokenize.py --mode 1 --input data/orign_data/1b.dev.txt --output data/process/pretrain.dev --stanford data/stanford-corenlp-full-2018-10-05

#head -n 1000000 data/process/pretrain.train > data/process/pretrain.train.1M
#head -n 20000 data/process/pretrain.dev > data/process/pretrain.dev.20K
#python scripts/genGraph.py --mode 1 --input data/process/pretrain.train.1M --output data/process/pretrain.train.1M.graph --process data/process/pretrain.train.1M.ic --stanford data/stanford-corenlp-full-2018-10-05
#python scripts/genGraph.py --mode 1 --input data/process/pretrain.dev.20K --output data/process/pretrain.dev.20K.graph --process data/process/pretrain.dev.20K.ic --stanford data/stanford-corenlp-full-2018-10-05

#python scripts/genVocab.py --mode 0 --input data/process/fce-public.train.preprocess.tsv data/process/fce-public.dev.preprocess.tsv --output data/prepare/wordvocab.pkl --mergeinput data/process/pretrain.train.1M.ic --mergemaxnum 50000
#python scripts/genVocab.py --mode 1 --input data/process/pretrain.train.1M.ic data/process/fce-public.train.preprocess.tsv data/process/fce-public.dev.preprocess.tsv --output data/prepare/charvocab.pkl
#python scripts/genVocab.py --mode 2 --input data/process/pretrain.train.1M.graph data/process/train_graph.txt data/process/dev_graph.txt --output data/prepare/edgevocab.pkl

#python scripts/binary.py --train-dir data/process/pretrain.train.1M.ic --dev-dir data/process/pretrain.dev.20K.ic --train-graph-dir data/process/pretrain.train.1M.graph --dev-graph-dir data/process/pretrain.dev.20K.graph \
#--word-vocab-dir data/prepare/wordvocab.pkl --char-vocab-dir data/prepare/charvocab.pkl --edge-vocab-dir data/prepare/edgevocab.pkl --output data/prepare/pretrain.pkl

#python scripts/binary.py --train-dir data/process/fce-public.train.preprocess.tsv --dev-dir data/process/fce-public.dev.preprocess.tsv --test-dir data/process/fce-public.test.preprocess.tsv \
#--train-graph-dir data/process/train_graph.txt --dev-graph-dir data/process/dev_graph.txt --test-graph-dir data/process/test_graph.txt \
#--word-vocab-dir data/prepare/wordvocab.pkl --char-vocab-dir data/prepare/charvocab.pkl --edge-vocab-dir data/prepare/edgevocab.pkl --output data/prepare/train.pkl

python main.py --gpu-list 0 --mode Train --arch GGNN --train-lm \
 --char-embed-dim 0 --gnn-steps 1 --save-dir checkpoint/LM_GGNN \
 --w2v-dir data/process/w2v_300d.txt --data-dir data/prepare/pretrain.pkl \
 --optimizer adam --lr 1e-3 --evaluation loss


# 预处理的预处理
# tokenize_()
# generate_graph(mode=0)

#binary

#pretrian

#train

#test
