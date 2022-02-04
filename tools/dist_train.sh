#!/usr/bin/env bash
set -e
set -x

CONFIG=$1
GPUS=$2
PORT=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir $PT_OUTPUT_DIR
