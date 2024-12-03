#!/bin/bash

export OPENAI_KEY=""

METHOD_1=$1

ANS_DIR_1=$2

OUTPUT_DIR_1=$3

ANS_DIR_2=$4

OUTPUT_DIR_2=$5

ANS_DIR_3=$6

OUTPUT_DIR_3=$7

ANS_DIR_4=$8

OUTPUT_DIR_4=$9

ANS_DIR_5=${10}

OUTPUT_DIR_5=${11}


python ./toolbench/tooleval/convert_to_answer_format.py --method $1 \
    --answer_dir $2 \
    --output $3

python ./toolbench/tooleval/convert_to_answer_format.py --method $1 \
    --answer_dir $4 \
    --output $5

python ./toolbench/tooleval/convert_to_answer_format.py --method $1 \
    --answer_dir $6 \
    --output $7

python ./toolbench/tooleval/convert_to_answer_format.py --method $1 \
    --answer_dir $8 \
    --output $9

python ./toolbench/tooleval/convert_to_answer_format.py --method $1 \
    --answer_dir ${10} \
    --output ${11}