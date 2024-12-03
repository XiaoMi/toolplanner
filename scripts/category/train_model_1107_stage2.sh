export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MODEL_PATH="ToolPlanner_Stage1_1020"
export SAVE_PATH="ToolPlanner_Stage2"
export DATA_PATH="data/category/dataset/G3_1107_gensample_Reward_pair.json"
export MASTER_ADDR="localhost"
export MASTER_PORT="20010"
export WANDB_DISABLED=true
wandb offline

torchrun --nproc_per_node=8 --master_port=20001 toolbench/train/train_long_seq_RRHF.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --gradient_checkpointing True \
    --tf32 True --model_max_length 8192 --rrhf_weight 1