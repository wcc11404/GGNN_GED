script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# tokenize标签数据
#python $script_dir/myscripts/mytokenize.py --mode 0 --input data/orign_data/fce-public.train.original.tsv --output data/process/fce-public.train.preprocess.tsv --stanford data/stanford-corenlp-full-2018-10-05
#python $script_dir/myscripts/mytokenize.py --mode 0 --input data/orign_data/fce-public.dev.original.tsv --output data/process/fce-public.dev.preprocess.tsv --stanford data/stanford-corenlp-full-2018-10-05
#python $script_dir/myscripts/mytokenize.py --mode 0 --input data/orign_data/fce-public.test.original.tsv --output data/process/fce-public.test.preprocess.tsv --stanford data/stanford-corenlp-full-2018-10-05

# 生成标签数据的依赖图
#python $script_dir/myscripts/genGraph.py --mode 0 --input data/process/fce-public.train.preprocess.tsv --output data/process/train_graph.txt --stanford data/stanford-corenlp-full-2018-10-05
#python $script_dir/myscripts/genGraph.py --mode 0 --input data/process/fce-public.dev.preprocess.tsv --output data/process/dev_graph.txt --stanford data/stanford-corenlp-full-2018-10-05
#python $script_dir/myscripts/genGraph.py --mode 0 --input data/process/fce-public.test.preprocess.tsv --output data/process/test_graph.txt --stanford data/stanford-corenlp-full-2018-10-05

# tokenize无标签数据
#python $script_dir/myscripts/mytokenize.py --mode 1 --input data/orign_data/1b.train.txt --output data/process/pretrain.train --stanford data/stanford-corenlp-full-2018-10-05
#python $script_dir/myscripts/mytokenize.py --mode 1 --input data/orign_data/1b.dev.txt --output data/process/pretrain.dev --stanford data/stanford-corenlp-full-2018-10-05

# 生成无标签数据的依赖图（部分）
#head -n 1000000 data/process/pretrain.train > data/process/pretrain.train.1M
#head -n 20000 data/process/pretrain.dev > data/process/pretrain.dev.20K
#python -u $script_dir/myscripts/corrupt.py --mode 1 --input data/process/pretrain.train.1M --output data/process/pretrain.train.1M.ic.error --errorrate 0.2 --dict-dir data/orign_data/fce-released-dataset/dataset
#python $script_dir/myscripts/corrupt.py --mode 1 --input data/process/pretrain.dev.20K --output data/process/pretrain.dev.20K.ic.error --errorrate 0.2 --dict-dir data/orign_data/fce-released-dataset/dataset
#python $script_dir/myscripts/genGraph.py --mode 0 --input data/process/pretrain.train.1M.ic.error --output data/process/pretrain.train.1M.graph --stanford data/stanford-corenlp-full-2018-10-05 --process data/process/pretrain.train.1M.ic.process
#python $script_dir/myscripts/genGraph.py --mode 0 --input data/process/pretrain.dev.20K.ic.error --output data/process/pretrain.dev.20K.graph --stanford data/stanford-corenlp-full-2018-10-05

#python $script_dir/myscripts/corrupt.py --input data/process/pretrain.train.2M.ic --output data/process/pretrain.train.2M.ic.error --errorrate 0.2 --mode 0
#python $script_dir/myscripts/corrupt.py --input data/process/pretrain.dev.40K.ic --output data/process/pretrain.dev.40K.ic.error --errorrate 0.2 --mode 0

# 生成word、char以及edge词典
#python $script_dir/myscripts/genVocab.py --mode 0 --input data/process/fce-public.train.preprocess.tsv data/process/fce-public.dev.preprocess.tsv --output data/prepare/wordvocab.pkl --mergeinput data/process/pretrain.train.1M.ic.process --mergemaxnum 50000
#python $script_dir/myscripts/genVocab.py --mode 1 --input data/process/pretrain.train.1M.ic.process data/process/fce-public.train.preprocess.tsv data/process/fce-public.dev.preprocess.tsv --output data/prepare/charvocab.pkl
#python $script_dir/myscripts/genVocab.py --mode 2 --input data/process/pretrain.train.1M.graph data/process/train_graph.txt data/process/dev_graph.txt --output data/prepare/edgevocab.pkl

# pickle化所有预训练数据
#python $script_dir/myscripts/binary.py --train-dir data/process/pretrain.train.2M.ic --dev-dir data/process/pretrain.dev.40K.ic --train-graph-dir data/process/pretrain.train.2M.graph --dev-graph-dir data/process/pretrain.dev.40K.graph \
# --word-vocab-dir data/prepare/wordvocab.pkl --char-vocab-dir data/prepare/charvocab.pkl --edge-vocab-dir data/prepare/edgevocab.pkl --output data/prepare/pretrain.pkl

# pickle化腐化预训练数据
#python $script_dir/myscripts/binary.py --train-dir data/process/pretrain.train.1M.ic.process --dev-dir data/process/fce-public.dev.preprocess.tsv \
# --train-graph-dir data/process/pretrain.train.1M.graph --dev-graph-dir data/process/dev_graph.txt \
# --word-vocab-dir data/prepare/wordvocab.pkl --char-vocab-dir data/prepare/charvocab.pkl --edge-vocab-dir data/prepare/edgevocab.pkl --output data/prepare/pretrain_corrupt.pkl

# pickle化所有训练数据
#python $script_dir/myscripts/binary.py --train-dir data/process/fce-public.train.preprocess.tsv --dev-dir data/process/fce-public.dev.preprocess.tsv --test-dir data/process/fce-public.test.preprocess.tsv \
# --train-graph-dir data/process/train_graph.txt --dev-graph-dir data/process/dev_graph.txt --test-graph-dir data/process/test_graph.txt \
# --word-vocab-dir data/prepare/wordvocab.pkl --char-vocab-dir data/prepare/charvocab.pkl --edge-vocab-dir data/prepare/edgevocab.pkl --output data/prepare/train.pkl

# 预训练
#python -u $script_dir/main.py --gpu-id 0 1 2 3 --mode Train --arch GGNNNER --criterion SLLoss \
# --char-embed-dim 0 --gnn-steps 1 --save-dir checkpoint/LM_GGNN_corrupt \
# --w2v-dir data/process/w2v_300d.txt --data-dir data/prepare/pretrain_corrupt.pkl \
# --optimizer adadelta --lr 1 --evaluation f0.5 --batch-size 16 --early-stop 5 \
# --max-epoch 15 --update-freq 1 --use-ddp

# fine-tune
#python -u $script_dir/main.py --gpu-id 3 --mode Train --arch GGNNNER --criterion SLLoss \
# --char-embed-dim 0 --gnn-steps 1 --save-dir checkpoint/GGNN_corrupt --load-dir checkpoint/LM_GGNN_corrupt \
# --data-dir data/prepare/train.pkl --optimizer adadelta --lr 1 --evaluation f0.5 \
# --batch-size 32 --early-stop 8 --max-epoch 50 --lm-cost-weight 0.10

# 训练
python -u $script_dir/main.py --gpu-id 1 --mode Train --arch GGNNNER --criterion SLLoss \
 --char-embed-dim 0 --gnn-steps 1 --save-dir checkpoint/paper_SL_GGNN --w2v-dir data/process/w2v_300d.txt \
 --data-dir data/prepare/train.pkl --optimizer adadelta --lr 1 --evaluation f0.5 \
 --batch-size 32 --early-stop 5 --max-epoch 15 --lm-cost-weight 0.10 --gnn-drop 0.3
