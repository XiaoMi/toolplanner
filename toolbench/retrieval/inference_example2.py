from sentence_transformers import SentenceTransformer, util
from toolbench.inference.LLM.retriever import ToolRetriever
import json
import pandas as pd
from collections import defaultdict
import torch
from tqdm import tqdm
import argparse
import os

# 创建参数解析器并添加参数
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, type=str, required=True, help='Your trained model path')
parser.add_argument('--dataset_path', help='The processed dataset files path')

corpus_tsv_path="data/retrieval/G3_clear/corpus.tsv"

output_file="data/test_sample/G3_query_100_opendomain_cut_retri_top20.json"
# 解析命令行参数
args = parser.parse_args()

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = args.model_path

retriever = ToolRetriever(corpus_tsv_path=corpus_tsv_path, model_path=model_path)




def retrieve_rapidapi_tools(query, top_k, jsons_path,gentools,retriflag):
    if retriflag==1:
        retrieved_tools = gentools+retriever.retrieving(query, top_k=top_k)
    else:
        retrieved_tools = gentools
    query_json = {"api_list":[]}
    for tool_dict in retrieved_tools:
        if len(query_json["api_list"]) == top_k:
            break
        category = tool_dict["category"]
        tool_name = tool_dict["tool_name"]
        api_name = tool_dict["api_name"]
        if os.path.exists(jsons_path):
            if os.path.exists(os.path.join(jsons_path, category)):
                if os.path.exists(os.path.join(jsons_path, category, tool_name+".json")):
                    query_json["api_list"].append({
                        "category_name": category,
                        "tool_name": tool_name,
                        "api_name": api_name
                    })
    return query_json
    

def build_tool_description(data_dict):
    white_list = get_white_list(tool_root_dir)
    origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
    tool_des = contain(origin_tool_names,white_list)
    tool_descriptions = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
    return tool_descriptions

def fetch_api_json(query_json):
    data_dict = {"api_list":[]}
    for item in query_json["api_list"]:
        cate_name = item["category_name"]
        tool_name = standardize(item["tool_name"])
        api_name = change_name(standardize(item["api_name"]))
        tool_json = json.load(open(os.path.join(tool_root_dir, cate_name, tool_name + ".json"), "r"))
        append_flag = False
        api_dict_names = []
        for api_dict in tool_json["api_list"]:
            api_dict_names.append(api_dict["name"])
            pure_api_name = change_name(standardize(api_dict["name"]))
            if pure_api_name != api_name:
                continue
            api_json = {}
            api_json["category_name"] = cate_name
            api_json["api_name"] = api_dict["name"]
            api_json["api_description"] = api_dict["description"]
            api_json["required_parameters"] = api_dict["required_parameters"]
            api_json["optional_parameters"] = api_dict["optional_parameters"]
            api_json["tool_name"] = tool_json["tool_name"]
            data_dict["api_list"].append(api_json)
            append_flag = True
            break
        if not append_flag:
            print(api_name, api_dict_names)
    return data_dict



querys = json.load(open(args.dataset_path, "r"))
test_list=[]

for query_id, data_dict in enumerate(querys):
    query=data_dict["query"]
    retrieved_api_nums=5
    tool_root_dir="data/toolenv/tools/"
    gentools=[]
    retriflag=1
    query_json = retrieve_rapidapi_tools(query, retrieved_api_nums, tool_root_dir,gentools,retriflag)

    print("#############################################")
    print(query_id)
    print(query)
    #print(query_json)
    category_list=[]
    tool_list=[]
    api_list=[]
    for temp_dict in query_json["api_list"]:
        if temp_dict["category_name"] not in category_list:
            category_list.append(temp_dict["category_name"])
        if temp_dict["tool_name"] not in tool_list:
            tool_list.append(temp_dict["tool_name"])
        if temp_dict["api_name"] not in api_list:
            api_list.append(temp_dict["api_name"])
    print(category_list)
    print(tool_list)
    print(api_list)
    print(len(category_list),len(tool_list),len(api_list))
    test_list.append({"query":query,"category_list":category_list,"tool_list":tool_list,"api_list":api_list})

#query_json = self.retrieve_rapidapi_tools(self.input_description, args.retrieved_api_nums, args.tool_root_dir,gentools,retriflag)
#data_dict = self.fetch_api_json(query_json)
#tool_descriptions = self.build_tool_description(data_dict)


json.dump(test_list, open(output_file,"w"), indent=4, ensure_ascii=False)  