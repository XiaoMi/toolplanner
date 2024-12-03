#!/bin/bash

export OPENAI_KEY=""

#METHOD_1=$1

#ANS_DIR_1=$2

OUTPUT_DIR_1=$1

#python ./toolbench/tooleval/convert_to_answer_format.py --method $1 \
#    --answer_dir $2 \
#    --output $3
python ./toolbench/tooleval/new_eval_win_rate_cut_list.py G3_instruction $1