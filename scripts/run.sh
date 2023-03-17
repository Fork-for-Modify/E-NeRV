set -x

CFG_PATH=$1
DIR=$2
PORT=$3
CUDA_VISIBLE_DEVICES=1

mkdir -p ./outputs/$DIR
TIME=$(date +"%Y%m%d_%H%M%S")

python -u main_orig.py \
    --cfg_path $CFG_PATH \
    --output_dir ./outputs/$DIR \
    --time_str $TIME \
    --port ${PORT} \
    2>&1 | tee ./outputs/$DIR/$TIME.log