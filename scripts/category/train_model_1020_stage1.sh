export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --master_port=20001 toolbench/train/train_long_seq.py \
    --model_name_or_path ToolLLaMA-7b  \
    --data_path  data/category/answer/G3_plan_gen_train_1020_G3_3tag_whole_prefixTagTraceAll.json \
    --eval_data_path  data/category/answer/G3_plan_gen_eval_1020_G3_3tag_whole_prefixTagTraceAll.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir ToolPlanner_Stage1 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --prediction_loss_only \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none