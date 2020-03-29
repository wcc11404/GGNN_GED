script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

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
 --char-embed-dim 0 --gnn-steps 1 --save-dir checkpoint/paper_SL_GGNN_7 --w2v-dir data/process/w2v_300d.txt \
 --data-dir data/prepare/train.pkl --optimizer adadelta --lr 1 --evaluation f0.5 \
 --batch-size 32 --early-stop 5 --max-epoch 15 --lm-cost-weight 0.15 --gnn-drop 0.0 --main-label-weight 1.2
