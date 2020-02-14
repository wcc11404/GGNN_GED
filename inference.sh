script_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# 测试
python $script_dir/main.py --gpu-id 0 --mode Test --arch GGNNNER --load-dir checkpoint/GGNN_att_sl \
 --data-dir data/prepare/train.pkl