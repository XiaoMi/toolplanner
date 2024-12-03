
DEVICE=$1

MODEL_DIR=$2

METHOD=$3

OUTPUT_DIR=$4

#Retri_Model_DIR=$5
INPUT_DIR=$5

export CUDA_VISIBLE_DEVICES=$1
export TOOLBENCH_KEY=$6

export INPUT_DIR=$5
export OUTPUT_DIR=$4
#export OUTPUT_DIR="data/answer/inference/plan_0925_cutcate_plan_retri"
export PYTHONPATH=./
#--retrieval_model_path retriever_model_G3_cut/2023-09-05_10-45-41/ \
#--retrieval_model_path retriever_model_G3_cut/2023-09-05_10-45-41/ \
#--retrieved_api_nums 5 \
#--corpus_tsv_path data/retrieval/G3_cut/corpus.tsv \
mkdir $OUTPUT_DIR
python toolbench/inference/qa_pipeline_open_domain.py \
    --tool_root_dir data/toolenv/tools/ \
    --corpus_tsv_path data/retrieval/G3_clear/corpus.tsv \
    --retrieval_model_path retriever_model_G3_clear/2023-09-14_21-00-57/ \
    --retrieved_api_nums 5 \
    --backbone_model toolllama \
    --model_path $2 \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method $3 \
    --input_query_file $INPUT_DIR \
    --output_answer_file $OUTPUT_DIR \
    --toolbench_key $TOOLBENCH_KEY

#cut retrieval
#--corpus_tsv_path data/retrieval/G3_cut/corpus.tsv \
#--retrieval_model_path retriever_model_G3_cut/2023-09-05_10-45-41/ \
#--retrieved_api_nums 5 \

#Cut_Cate_Tool_CutToolPlan_FullToolRetri_CutToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_FullToolRetri_FullToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_FullRetri_CutToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_FullRetri_FullToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_CutToolRetri_CutToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_CutToolRetri_FullToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_CutRetri_CutToolGen_DFS_woFilter_w2
#Cut_Cate_Tool_CutToolPlan_CutRetri_FullToolGen_DFS_woFilter_w2
#plan_1012_G3_cut2tool_fulltoolplan_FullToolRetri_CutToolGen
#plan_1012_G3_cut2tool_fulltoolplan_FullToolRetri_FullToolGen
#plan_1012_G3_cut2tool_fulltoolplan_FullRetri_CutToolGen
#plan_1012_G3_cut2tool_fulltoolplan_FullRetri_FullToolGen 
#plan_1012_G3_cut2tool_fulltoolplan_CutToolRetri_CutToolGen
#plan_1012_G3_cut2tool_fulltoolplan_CutToolRetri_FullToolGen
#plan_1012_G3_cut2tool_fulltoolplan_CutRetri_CutToolGen
#plan_1012_G3_cut2tool_fulltoolplan_CutRetri_FullToolGen