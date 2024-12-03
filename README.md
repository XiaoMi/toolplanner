ToolPlanner
===========================

## Paper Link
[ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback](https://arxiv.org/abs/2409.14826)

****
## 目录
* [Requirement](##Requirement)
* [Data](##Data)
* [Model](#Model)


## Requirement

```
accelerate==0.24.0
datasets==2.13.0
deepspeed==0.9.2
Flask==1.1.2
Flask_Cors==4.0.0
huggingface_hub==0.16.4
jsonlines==3.1.0
nltk==3.7
numpy==1.24.3
openai==0.27.7
pandas==2.0.3
peft==0.6.0.dev0
psutil==5.8.0
pydantic==1.10.8
pygraphviz==1.11
PyYAML==6.0
PyYAML==6.0.1
Requests==2.31.0
scikit_learn==1.0.2
scipy==1.11.4
sentence_transformers==2.2.2
tenacity==8.2.3
termcolor==2.4.0
torch==2.0.1
tqdm==4.65.0
transformers==4.28.1
trl==0.7.3.dev0
```

## Data

|path|data description|
|----|-----|
|[/data/category/dataset]|MGToolBench: pairwise_responses|
|[/data/category/answer](./data/category/answer)|MGToolBench: Multi-Level Instruction Split|
|[/data/category/coarse_instruction](./data/category/coarse_instruction)|Self-Instruct Data: multi-granularity instructions|
|[/data/test_sample](./data/test_sample)|Test Sample: test dataset|
|[/data/category/toolenv]|Tool Environment: Tools, APIs, and their documentation.|
|[/data/category/inference]|Output: solution trees path|
|[/data/category/converted_answer](./data/category/converted_answer)|Output: converted_answer path|
|[/data/category/retrieval/G3_category](./data/category/retrieval/G3_category)|Supplementary: Category & Tool & API Name|
|[/data/retrieval/G3_clear](./data/retrieval/G3_clear)|Supplementary: corpus for seperate retriever|

## Download Data and Checkpoints

download these data and unzip them.
|path|data description|data name|url|
|----|-----|-----|-----|
|[/data/category/answer]|MGToolBench: sft training dataset|G3_plan_gen_train_1020_G3_3tag_whole_prefixTagTraceAll.json|https://huggingface.co/datasets/wuqinzhuo/ToolPlanner|
|[/data/category/dataset]|MGToolBench: pairwise_responses|G3_1107_gensample_Reward_pair.json|https://huggingface.co/datasets/wuqinzhuo/ToolPlanner|
|[/data/category/toolenv]|Tool Environment: Tools, APIs, and their documentation.|toolenv.zip|https://huggingface.co/datasets/wuqinzhuo/ToolPlanner|
|[/data/category/inference]|Output: solution trees path|inference.zip|https://huggingface.co/datasets/wuqinzhuo/ToolPlanner|
|[/data/retrieval/G3_clear]|Training dataset for Retrivel model|train.json|https://huggingface.co/datasets/wuqinzhuo/ToolPlanner|
|[/data/retrieval/G3_clear]|Training dataset for Retrivel mode|corpus.tsv|https://huggingface.co/datasets/wuqinzhuo/ToolPlanner|


|path|model description|model name|url|
|----|-----|-----|-----|
|[ToolPlanner root path]|Stage1 sft model|ToolPlanner_Stage1_1020|https://huggingface.co/wuqinzhuo/ToolPlanner_Stage1_1020|
|[ToolPlanner root path]|Stage1 sft model|ToolPlanner_Stage2_1107|https://huggingface.co/wuqinzhuo/ToolPlanner_Stage2_1107/|
|[ToolPlanner root path]|Baseline ToolLLaMA|ToolLLaMA-7b|https://github.com/OpenBMB/ToolBench|
|[ToolPlanner root path]|Retrivel model for test, using MGToolBench data|model_1122_G3_tag_trace_multilevel|https://huggingface.co/wuqinzhuo/model_1122_G3_tag_trace_multilevel|
|[ToolPlanner root path]|Retrivel model for test, using ToolBench data|retriever_model_G3_clear|https://huggingface.co/wuqinzhuo/retriever_model_G3_clear|


# Model
## Install
    pip install -r requirements.txt


## Train ToolPlanner, Stage 1 SFT
### Script
    bash scripts/category/train_model_1020_stage1.sh 
### Code
```
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
```

## Train ToolPlanner, Stage 2 Reinforcement Learning
### Script
    bash scripts/category/train_model_1107_stage2.sh 
### Code
```
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
```

## Inference, Generate Solution Tree
### Script
```
bash scripts/category/inference/inference_cuda_model_method_output_input_tag.sh <GPU_Id> <model_name> <method_name> <decode_method> <output_path> <test_sample> <retriever_path> <TOOLBENCH_KEY>
```

### ToolBench Key
Go to [ToolBench](https://github.com/OpenBMB/ToolBench) to apply for a [ToolBench Key](https://github.com/OpenBMB/ToolBench). 


### Decode_Method

|Model|Method|
|----|-----|
|`Full Model`|`Mix_Whole3Tag_MixWhole3TagTrace_3TagRepla_PureRepla_MixWhole3Retri_MixWhole3TagTraceGen_DFS_woFilter_w2`|
|`Seperate Retriever`|`Mix_Whole3Tag_MixWhole3TagTrace_MixWhole3Retri_MixWhole3TagTraceGen_DFS_woFilter_w2`|
|`Without Solution Planning`|`Mix_Whole3Tag_MixWhole3TagTrace_MixWhole3Retri_MixWhole3Gen_DFS_woFilter_w2`|
|`Without Tag Extraction`|`Mix_Whole3Tag_MixWhole3TagTrace_MixTagTraceRetri_MixTagTraceGen_DFS_woFilter_w2`|
|`Without Tag & Solution`|`Mix_Whole3Tag_MixWhole3TagTrace_MixRetri_MixGen_DFS_woFilter_w2`|
|`Chain-based Method`|`Mix_Whole3Tag_MixWhole3TagTrace_3TagRepla_PureRepla_MixWhole3Retri_MixWhole3TagTraceGen_CoT@5`|


### Example
```
bash scripts/category/inference/inference_cuda_model_method_output_input_tag.sh 6,7 ToolPlanner_Stage2_1107 Mix_Whole3Tag_MixWhole3TagTrace_3TagRepla_PureRepla_MixWhole3Retri_MixWhole3TagTraceGen_DFS_woFilter_w2 data/category/inference/plan_1107_G3_gensample_RRHF_Desc_1122_level_23 data/test_sample/G3_query_100_opendomain.json model_1122_G3_tag_trace_multilevel TOOLBENCH_KEY

bash scripts/category/inference/inference_cuda_model_method_output_input_tag.sh 1,3 ToolPlanner_Stage2_1107 Mix_Whole3Tag_MixWhole3TagTrace_3TagRepla_PureRepla_MixWhole3Retri_MixWhole3TagTraceGen_DFS_woFilter_w2 data/category/inference/plan_1107_G3_gensample_RRHF_Cate_1122_level_23 data/test_sample/G3_query_100_level_cate.json model_1122_G3_tag_trace_multilevel TOOLBENCH_KEY
bash scripts/category/inference/inference_cuda_model_method_output_input_tag.sh 2,4 ToolPlanner_Stage2_1107 Mix_Whole3Tag_MixWhole3TagTrace_3TagRepla_PureRepla_MixWhole3Retri_MixWhole3TagTraceGen_DFS_woFilter_w2 data/category/inference/plan_1107_G3_gensample_RRHF_Tool_1122_level_23 data/test_sample/G3_query_100_level_tool.json model_1122_G3_tag_trace_multilevel TOOLBENCH_KEY
bash scripts/category/inference/inference_cuda_model_method_output_input_tag.sh 5,4 ToolPlanner_Stage2_1107 Mix_Whole3Tag_MixWhole3TagTrace_3TagRepla_PureRepla_MixWhole3Retri_MixWhole3TagTraceGen_DFS_woFilter_w2 data/category/inference/plan_1107_G3_gensample_RRHF_API_1122_level_23 data/test_sample/G3_query_100_level_api.json model_1122_G3_tag_trace_multilevel TOOLBENCH_KEY
```

## Eval
### Script
Use generated results to eval Match Rate and Pass Rate
```
bash scripts/category/eval/eval_match_pass_rate.sh api name2 <output_path>
```

### Example
```
bash scripts/category/eval/eval_match_pass_rate.sh api name2 data/category/inference/plan_1107_G3_gensample_RRHF_Cate_1122_level_23
bash scripts/category/eval/eval_match_pass_rate.sh api name2 data/category/inference/plan_1107_G3_gensample_RRHF_Tool_1122_level_23
bash scripts/category/eval/eval_match_pass_rate.sh api name2 data/category/inference/plan_1107_G3_gensample_RRHF_API_1122_level_23
bash scripts/category/eval/eval_match_pass_rate.sh api name2 data/category/inference/plan_1107_G3_gensample_RRHF_Desc_1122_level_23
```

### Script
Use generated results to eval Win Rate
```
Change generate(prompt, name) function in "ToolPlanner/toolbench/tooleval/new_eval_win_rate_cut_list.py" to your own ChatGPT API.

bash scripts/category/eval/eval_match_pass_rate.sh api name2 <output_path>
```

### Example
```
bash scripts/inference/convert_preprocess_win_rate.sh DFS data/category/inference/plan_1107_G3_gensample_RRHF_Cate_1122_level_23 data/category/converted_answer/plan_1107_G3_gensample_RRHF_Cate_1122_level_23.json data/category/inference/plan_1107_G3_gensample_RRHF_Tool_1122_level_23 data/category/converted_answer/plan_1107_G3_gensample_RRHF_Tool_1122_level_23.json data/category/inference/plan_1107_G3_gensample_RRHF_API_1122_level_23 data/category/converted_answer/plan_1107_G3_gensample_RRHF_API_1122_level_23.json data/category/inference/plan_1107_G3_gensample_RRHF_Desc_1122_level_23 data/category/converted_answer/plan_1107_G3_gensample_RRHF_Desc_1122_level_23.json
bash scripts/inference/eval_win_rate_cut_list.sh data/category/converted_answer/plan_1107_G3_gensample_RRHF_Cate_1122_level_23.json
```

### Citation
```
@misc{wu2024toolplannertoolaugmentedllm,
      title={ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback}, 
      author={Qinzhuo Wu and Wei Liu and Jian Luan and Bin Wang},
      year={2024},
      eprint={2409.14826},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.14826}, 
}
```

### License

The dataset of this project is licensed under the [**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The source code of the this is licensed under the [**Apache 2.0**](http://www.apache.org/licenses/LICENSE-2.0)  license.

#### Summary of Terms
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial**: You may not use the material for commercial purposes.
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.


#### License Badge
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### 5. Citation
If you'd like to use our benchmark or cite this paper, please kindly use the reference below:

```bibtex
@inproceedings{wu2024toolplanner,
  title={ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback},
  author={Wu, Qinzhuo and Liu, Wei and Luan, Jian and Wang, Bin},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={18315--18339},
  year={2024}
}

