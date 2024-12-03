# Copyright (C) 2024 Xiaomi Corporation.
# The source code included in this project is licensed under the Apache 2.0 license.
import os
import re
import json
import numpy as np
import sys
import argparse
from sentence_transformers import SentenceTransformer, util
from toolbench.inference.LLM.retriever import ToolRetriever
import torch

#parser = argparse.ArgumentParser()
#parser.add_argument('--answer_dir',type=str, required=True,help='where the answers stored.')

ground_dir = sys.argv[1]
input_dir = sys.argv[2]
method = sys.argv[3]


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def standardize_api(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    #if string[0].isdigit():
    #    string = "get_" + string
    return string


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

long_replace_pair={}
#count cut dulpicate
cut_list=[]
cate_list=[]
tool_list=[]
full_list=[]
categoryName={}
toolName={}
APIName={}

cate2api={}
tool2api={}
cate2tool={}
sortCateLine=""
tempfile=open("data/category/retrieval/G3_category/cateNameStand.txt", "r")
for templine in tempfile:
    key=templine.strip()
    if len(key)>1:
        sortCateLine=key
print("#######################self.sortCateLine######################")
print(sortCateLine)
tempfile=open("data/category/retrieval/G3_category/toolNameStand.txt", "r")
for templine in tempfile:
    key=templine.strip().split("\t")
    toolName[key[0]]=key[1]
    tool_list=key[1].strip().strip(".").split(", ")
    cate2tool[standardize_category(key[0])]=[standardize(tool) for tool in tool_list]
print("#######################self.toolname######################")
#print(toolName)
print(len(cate2tool))



tempfile=open("data/category/retrieval/G3_category/APINameStand.txt", "r").readlines()
print(len(tempfile))
count=0
for templine in tempfile:
    count+=1
    key=templine.strip().split("\t")
    APIName[key[0]]=key[1]
    api_list=key[1].strip().strip(".").split(", ")
    tool=standardize(key[0])
    assert tool not in tool2api
    tool2api[tool]=[change_name(standardize(api)) for api in api_list]
print("#######################self.APIName######################")
#print(APIName)
print(count)
print(len(tool2api))

for key in cate2tool:
    temp_apiList=[]
    for key1 in cate2tool[key]:
        if key1 not in tool2api:
            print(key1)
            print(cate2tool[key])
            print(tool2api.keys())
        temp_apiList.extend(tool2api[key1])
    cate2api[key]=temp_apiList
'''
def dfs(node, func):
    S = set()
    if node['node_type'] == 'Action':
        name = node['description']
        flag = False
        for f in func:
            if f.endswith(name):
                S.add(f)
    for child in node['children']:
        T = dfs(child, func)
        S = S.union(T)
    return S
'''
def dfs(node, func):
    S = set()
    if node['node_type'] == 'Action':
        name = node['description']
        flag = False
        if len(node['children'])>0:
            if node['children'][0]['node_type'] == 'Action Input':
                if "No such function name" not in node['children'][0]['observation']:
                    S.add(name)
    for child in node['children']:
        T = dfs(child, func)
        S = S.union(T)
    return S

def left_dfs(node, func):
    S = set()
    if node['node_type'] == 'Action':
        name = node['description']
        flag = False
        if len(node['children'])>0:
            if node['children'][0]['node_type'] == 'Action Input':
                if "No such function name" not in node['children'][0]['observation']:
                    S.add(name)

    #for child in node['children']:
    if len(node['children'])>0:
        child=node['children'][0]
        T = left_dfs(child, func)
        S = S.union(T)
    return S


def intersection(l1,l2):
    l3=l1[:]
    hit=[]
    for word in l2:
        if word in l3:
            index=l3.index(word)
            l3[index]="############"
            hit.append(word)
    return hit

def check_real_valid(string):
    fake_true_vocab = ["sorry","apologize","apology","unfortunately","couldn't"]
    for word in fake_true_vocab:
        if word in string.lower():
            return False
    return True


def check_pass_flag(query,function_step_list,target_step_list):
    pass_flag=False
    if function_step_list[-1]=="Finish":
        #print("****####***####")
        #print(target_step_list[-1])
        if '"return_type": "give_answer"' in target_step_list[-1]:
            pass_flag= check_real_valid(target_step_list[-1].split('"final_answer": "')[1])
    return pass_flag


def check_match_flag(tagmessage,function_step_list):
    match_flag=[False,False,False,False,False,False]
    correct_flag=True
    ground_cate=tagmessage.split("\nCate_Tag: ")[1].split("\n")[0].strip(".").split(", ")
    if "\nTool_Tag: " not in tagmessage:
        ground_tool=[]
    else:
        ground_tool=tagmessage.split("\nTool_Tag: ")[1].split("\n")[0].strip(".").split(", ")
    if "\nAPI_Tag: " not in tagmessage:
        ground_api=[]
    else:
        ground_api=tagmessage.split("\nAPI_Tag: ")[1].split("\n")[0].strip(".").split(", ")
    gen_cate=[]
    gen_tool=[]
    gen_api=[]
    temp_flag=False
    print("###########check_match_flag")
    print("function_step_list",function_step_list)

    for apifortool in function_step_list:
        if "_for_" in apifortool:
            tool_trace=apifortool.split("_for_")[-1]
            api_trace="_for_".join(apifortool.split("_for_")[0:-1])
            if "mobile_apps"== tool_trace:
                tool_trace="_for_".join(apifortool.split("_for_")[-2:])
                api_trace="_for_".join(apifortool.split("_for_")[0:-2])
            if "for_check_username"== tool_trace:
                tool_trace="check_username"
                api_trace=apifortool.split("_for_check_username")[0]
            if tool_trace not in tool2api:
                tool_trace1="_for_".join(apifortool.split("_for_")[-2:])
                if tool_trace1 in tool2api:
                    tool_trace="_for_".join(apifortool.split("_for_")[-2:])
                    api_trace="_for_".join(apifortool.split("_for_")[0:-2])
            if len(api_trace)<1:
                continue
            if len(apifortool)>50:
                if tool_trace not in tool2api:
                    correct_flag=False
                    continue
                if api_trace not in tool2api[tool_trace]:
                    if api_trace in long_replace_pair:
                        api_trace=long_replace_pair[api_trace]
                    else:
                        for temp_name in tool2api[tool_trace]:
                            if api_trace in temp_name:
                                long_replace_pair[api_trace]=temp_name
                                api_trace=temp_name
                    temp_flag=True
            tool_trace=standardize(tool_trace)
            api_trace=standardize(api_trace)
            cate_trace=""
            cate_trace=""
            for cate in cate2tool:
                if tool_trace in cate2tool[cate]:
                    cate_trace=cate
                    break
            if tool_trace not in tool2api:
                #wrong_tool2api_dict[data_file]=["tool_trace wrong",apifortool,tool_trace]
                correct_flag=False
                continue

            if api_trace not in tool2api[tool_trace]:
                #wrong_tool2api_dict[data_file]=["api_trace wrong",apifortool,tool2api[tool_trace]]
                #for temp_name in tool2api[tool_trace]:
                #    if api_trace in temp_name:
                #        function_name_replace_list[api_trace]=[temp_name,tool_trace,apifortool,temp_name+"_for_"+tool_trace]
                correct_flag=False
                print("api_trace",api_trace)
                print("tool2api[tool_trace]",tool2api[tool_trace])
                #return match_flag,correct_flag

            #assert api_trace in  tool2api[tool_trace] 
            #print(cate2api.keys())
            #assert api_trace in  cate2api[cate_trace] 
            gen_cate.append(cate_trace)
            gen_tool.append(tool_trace)
            gen_api.append(api_trace)

    ground_cate=set(ground_cate)
    ground_tool=set(ground_tool)
    ground_api=set(ground_api)
    gen_cate=set(gen_cate)
    gen_tool=set(gen_tool)
    gen_api=set(gen_api)

    a1=[0,0,0,0]
    a2=[0,0,0,0]
    a3=[0,0,0,0]

    for temp in gen_cate:
        if temp in ground_cate:
            a1[0]+=1
        else:
            a1[1]+=1
    for temp in ground_cate:
        if temp in gen_cate:
            a1[2]+=1
        else:
            a1[3]+=1
    for temp in gen_tool:
        if temp in ground_tool:
            a2[0]+=1
        else:
            a2[1]+=1
    for temp in ground_tool:
        if temp in gen_tool:
            a2[2]+=1
        else:
            a2[3]+=1
    for temp in gen_api:
        if temp in ground_api:
            a3[0]+=1
        else:
            a3[1]+=1
    for temp in ground_api:
        if temp in gen_api:
            a3[2]+=1
        else:
            a3[3]+=1

    #print("**********************")
    #print(ground_cate)
    #print(ground_tool)
    #print(ground_api)
    #print(gen_cate)
    #print(gen_tool)
    #print(gen_api)
    #print("**********************")
    match_flag=[]
    
    if gen_cate==ground_cate:
        match_flag.append(True)
    else:
        match_flag.append(False)
    #if gen_cate.issubset(ground_cate)and len(gen_cate)>0:
    #    match_flag.append(True)
    #else:
    #    match_flag.append(False)
    match_flag.append(a1)
    if gen_tool==ground_tool:
        match_flag.append(True)
    else:
        match_flag.append(False)
    #if gen_tool.issubset(ground_tool)and len(gen_tool)>0:
    #    match_flag.append(True)
    #else:
    #    match_flag.append(False)
    
    match_flag.append(a2)
    if gen_api==ground_api:
        match_flag.append(True)
    else:
        match_flag.append(False)
    #if gen_api.issubset(ground_api) and len(gen_api)>0:
    #    match_flag.append(True)
    #else:
    #    match_flag.append(False)

    match_flag.append(a3)


    return match_flag,correct_flag


def pair_posi_nega_data(out_list,gen_out_list,out_list_nega,gen_out_list_nega):
    dict_pair={}
    count_pair_num=0
    print("############################")
    for tmp_instances in out_list:
        #print(tmp_instances.keys())
        query_id=tmp_instances["query_id"].split("_")[0]
        input_type=tmp_instances["input_type"]
        query=tmp_instances["conversations"][1]["value"].strip().split("Cate_Tag")[0]
        step=tmp_instances["id"].split(":")[0]
        key=query_id+"_"+input_type
        #key=step+"###"+query
        if key not in dict_pair:
            tmp_instances["query"]=query
            tmp_instances["conversations"]=[tmp_instances["conversations"]]
            tmp_instances["reward"]=[tmp_instances["reward"]]
            dict_pair[key]=tmp_instances
        else:
            count_pair_num+=1
            if query != dict_pair[key]["query"]:
                print(key)
                print(dict_pair[key]["conversations"][0][1]["value"])
                print(query)
                print("**************************************")
            dict_pair[key]["conversations"].append(tmp_instances["conversations"])
            dict_pair[key]["reward"].append(tmp_instances["reward"])
            if count_pair_num%1000==0:
                print(dict_pair[key]["reward"])
    print("len out_list",len(out_list))
    print("count_pair_num",count_pair_num)

    for tmp_instances in gen_out_list:
        query_id=tmp_instances["query_id"].split("_")[0]
        input_type=tmp_instances["input_type"]
        query=tmp_instances["conversations"][1]["value"].strip().split("Cate_Tag")[0]
        step=tmp_instances["id"].split(":")[0]
        key=query_id+"_"+input_type
        #key=step+"###"+query
        if key not in dict_pair:
            tmp_instances["query"]=query
            tmp_instances["conversations"]=[tmp_instances["conversations"]]
            tmp_instances["reward"]=[tmp_instances["reward"]]
            dict_pair[key]=tmp_instances
        else:
            count_pair_num+=1
            if query != dict_pair[key]["query"]:
                print(key)
                print(dict_pair[key]["query"])
                print(query)
                print("**************************************")
            dict_pair[key]["conversations"].append(tmp_instances["conversations"])
            dict_pair[key]["reward"].append(tmp_instances["reward"])
            if count_pair_num%1000==0:
                print(dict_pair[key]["reward"])
    print("len gen_out_list",len(gen_out_list))
    print("count_pair_num",count_pair_num)
    for tmp_instances in out_list_nega:
        query_id=tmp_instances["query_id"].split("_")[0]
        input_type=tmp_instances["input_type"]
        query=tmp_instances["conversations"][1]["value"].strip().split("Cate_Tag")[0]
        step=tmp_instances["id"].split(":")[0]
        key=query_id+"_"+input_type
        #key=step+"###"+query
        if key not in dict_pair:
            tmp_instances["query"]=query
            tmp_instances["conversations"]=[tmp_instances["conversations"]]
            tmp_instances["reward"]=[tmp_instances["reward"]]
            dict_pair[key]=tmp_instances
        else:
            count_pair_num+=1
            if query != dict_pair[key]["query"]:
                print(key)
                print(dict_pair[key]["query"])
                print(query)
                print("**************************************")
            dict_pair[key]["conversations"].append(tmp_instances["conversations"])
            dict_pair[key]["reward"].append(tmp_instances["reward"])
            if count_pair_num%1000==0:
                print(dict_pair[key]["reward"])
    print("len gen_out_list",len(out_list_nega))
    print("count_pair_num",count_pair_num)
    for tmp_instances in gen_out_list_nega:
        query_id=tmp_instances["query_id"].split("_")[0]
        input_type=tmp_instances["input_type"]
        query=tmp_instances["conversations"][1]["value"].strip().split("Cate_Tag")[0]
        step=tmp_instances["id"].split(":")[0]
        key=query_id+"_"+input_type
        #key=step+"###"+query
        if key not in dict_pair:
            tmp_instances["query"]=query
            tmp_instances["conversations"]=[tmp_instances["conversations"]]
            tmp_instances["reward"]=[tmp_instances["reward"]]
            dict_pair[key]=tmp_instances
        else:
            count_pair_num+=1
            if query != dict_pair[key]["query"]:
                print(key)
                print(dict_pair[key]["query"])
                print(query)
                print("**************************************")
            dict_pair[key]["conversations"].append(tmp_instances["conversations"])
            dict_pair[key]["reward"].append(tmp_instances["reward"])
            if count_pair_num%1000==0:
                print(dict_pair[key]["reward"])
    print("len gen_out_list",len(gen_out_list_nega))
    print("count_pair_num",count_pair_num)

    pair_posi_nega=[]
    query_id_list=[]
    query_type_list=[]
    query_type_dict={"cate":0,"tool":0,"api":0,"desc":0}
    count10=0
    count01=0
    count00=0
    count_num=0
    for key in dict_pair:
        if len(dict_pair[key]["reward"])>1:
            flag1=0
            flag2=0
            for x in dict_pair[key]["reward"]:
                if x==1.0:
                    flag1=1
                if x<0:
                    flag2=1
            if flag1==1 and flag2==1:
                pair_posi_nega.append(dict_pair[key])
                query_id_list.append(key.split("_")[0])
                query_type_list.append(key.split("_")[0]+"_"+key.split("_")[1])
                count_num+=len(dict_pair[key]["reward"])
                if "cate" in key:
                    query_type_dict["cate"]+=1
                if "tool" in key:
                    query_type_dict["tool"]+=1
                if "api" in key:
                    query_type_dict["api"]+=1
                if "desc" in key:
                    query_type_dict["desc"]+=1
            if flag1==1 and flag2==0:
                count10+=1
            if flag1==0 and flag2==1:
                count01+=1
            if flag1==0 and flag2==0:
                count00+=1
    print(len(query_id_list))
    print(len(set(query_id_list)))
    print(len(set(query_type_list)))
    print("count10",count10)
    print("count01",count01)
    print("count00",count00)
    print("count_num",count_num)
    print(query_type_dict)
    print(set(query_id_list))
    return pair_posi_nega


if __name__=='__main__':
    test_count = "100"
    method2result = {}
    method2querycount = {}

    fin = open(ground_dir, 'r', encoding='utf-8')
    G = json.loads(fin.read())
    fin.close()
    GG = {}
    GList={}
    for x in G:
        GG[x['query_id']] = x['relevant APIs']
        GList[x['query_id']] = x['api_list']



    tempfile=json.load(open("data/test_sample/G3_query_100_opendomain_cut_retri_top20.json", "r"))
    data_retri_top20={}
    
    for query_id, data_dict in enumerate(tempfile):
        query=data_dict["query"]
        category_list=data_dict["category_list"]
        tool_list=data_dict["tool_list"]
        api_list=data_dict["api_list"]
        data_retri_top20[query]=[category_list,tool_list,api_list]


    corpus_tsv_path="data/retrieval/G3_clear/corpus.tsv"
    #model_path="retriever_model_G3_clear/2023-09-14_21-00-57/"
    #model_path="retriever_G3_mix3tag_1024/2023-10-24//"
    model_path="retriever_G3_mix3tag_1202/2023-12-02_15-13-22/"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    retriever = ToolRetriever(corpus_tsv_path=corpus_tsv_path, model_path=model_path)


    def retrieve_rapidapi_tools(query, top_k, jsons_path,gentools,retriflag):
        if retriflag==1:
            retrieved_tools = gentools+retriever.retrieving(query, top_k=top_k)
        else:
            retrieved_tools = gentools
        query_json = []
        for tool_dict in retrieved_tools:
            if len(query_json) == top_k:
                break
            query_json.append(tool_dict)


        return query_json
    total_hit = 0.0
    total_gold = 0.0
    total_pred = 0.0

    cate_hit=0.0
    cate_gold=0.0
    cate_pred=0.0
    cate_total=0.0
    cate_covergold=0.0
    cate_coverpred=0.0

    tool_hit=0.0
    tool_gold=0.0
    tool_pred=0.0
    tool_total=0.0
    tool_covergold=0.0
    tool_coverpred=0.0


    api_hit=0.0
    api_gold=0.0
    api_pred=0.0
    api_total=0.0
    api_covergold=0.0
    api_coverpred=0.0

    total_case=0.0
    availble=0.0
    total_steps=0.0
    total_use_functions=0.0


    cate_retri_hit=0.0
    cate_retri_total=0.0
    cate_retri_allhit=0.0
    tool_retri_hit=0.0
    tool_retri_total=0.0
    tool_retri_allhit=0.0
    api_retri_hit=0.0
    api_retri_total=0.0
    api_retri_allhit=0.0

    pass_num=0.0

    count_pass=0.0
    count_cate_match=0.0
    count_tool_match=0.0
    count_api_match=0.0


    count_cate_match_dfs=0.0
    count_tool_match_dfs=0.0
    count_api_match_dfs=0.0
    count_cate_match_dfsleft=0.0
    count_tool_match_dfsleft=0.0
    count_api_match_dfsleft=0.0

    count_tag_cate=0.0
    count_tag_tool=0.0
    count_tag_api=0.0
    count_trace_cate=0.0
    count_trace_tool=0.0
    count_trace_api=0.0

    count_0_cate=0
    count_1_cate=0
    count_0_api=0
    count_1_api=0
    count_0_tool=0
    count_1_tool=0

    count_2_cate=0
    count_3_cate=0
    count_2_api=0
    count_3_api=0
    count_2_tool=0
    count_3_tool=0

    for f in os.listdir(input_dir):
        query_id = int(f.split('_')[0])

        gold = set()
        category_list=[]
        tool_list=[]
        api_list=[]
        for x in GG[query_id]:
            tool_name = x[0]
            api_name = x[1]
            for api in GList[query_id]:
                if api['tool_name']==tool_name:
                    category_name=api['category_name']
                    break
            tool_name = standardize(tool_name)
            tool_list.append(tool_name)
            if len(category_name)>0:
                category_name = standardize_category(category_name)
                category_list.append(category_name)
            api_name = change_name(standardize(api_name))
            api_list.append(api_name)

        catelist1=[]
        for cate in category_list:
            cate=standardize_category(cate)
            catelist1.append(cate)
        catenames=", ".join(catelist1)
        toollist1=[]
        for tool in tool_list:
            tool = standardize(tool)
            toollist1.append(tool)
        toolnames=", ".join(toollist1)
        apilist1=[]
        toollist1=[]
        for api in api_list:
            api=change_name(standardize(api))
            apilist1.append(api)
        apinames=", ".join(apilist1)

        tagmessage="\nCate_Tag: "+catenames+"."
        tagmessage=tagmessage+"\nTool_Tag: "+toolnames+"."
        tagmessage=tagmessage+"\nAPI_Tag: "+apinames+"."


        fin = open('%s/%s' % (input_dir, f), 'r', encoding='utf-8')
        J = json.loads(fin.read())
        fin.close()
        func = set()
        for x in J['answer_generation']['function']:
            name = x['name']
            if name != 'Finish':
                func.add(name)
        #root = J['tree']['tree']
        #pred = dfs(root, func)
        #print(func)
        print(tagmessage)

        query= J['answer_generation']['query']
        retrieved_api_nums=5
        tool_root_dir="data/toolenv/tools/"
        gentools=[]
        retriflag=1
        retrieved_tools = retrieve_rapidapi_tools(query, retrieved_api_nums, tool_root_dir,gentools,retriflag)
        func1=[]
        for tool_dict in retrieved_tools:
            func1.append(tool_dict["api_name"]+"_for_"+tool_dict["tool_name"])
        print(retrieved_tools)
        print(func1)
        match_flag,correct_flag=check_match_flag(tagmessage,func1)
        print(match_flag)
        if match_flag[0]==True:
            count_tag_cate+=1
        if match_flag[2]==True:
            count_tag_tool+=1
        if match_flag[4]==True:
            count_tag_api+=1

        count_0_cate+=match_flag[1][0]
        count_1_cate+=match_flag[1][1]
        count_0_tool+=match_flag[3][0]
        count_1_tool+=match_flag[3][1]
        count_0_api+=match_flag[5][0]
        count_1_api+=match_flag[5][1]
        count_2_cate+=match_flag[1][2]
        count_3_cate+=match_flag[1][3]
        count_2_tool+=match_flag[3][2]
        count_3_tool+=match_flag[3][3]
        count_2_api+=match_flag[5][2]
        count_3_api+=match_flag[5][3]



    print("%d\t%d\t%d"% (count_tag_cate, count_tag_tool, count_tag_api))
    P1=float(count_0_cate)/(count_0_cate+count_1_cate)
    P2=float(count_0_tool)/(count_0_tool+count_1_tool)
    P3=float(count_0_api)/( count_0_api+count_1_api)
    R1=float(count_2_cate)/(count_2_cate+count_3_cate)
    R2=float(count_2_tool)/(count_2_tool+count_3_tool)
    R3=float(count_2_api)/(count_2_api+count_3_api)
    F1=(2*P1*R1)/(P1+R1)
    F2=(2*P2*R2)/(P2+R2)
    F3=(2*P3*R3)/(P3+R3)
    print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f"% (P1, P2,P3,R1,R2,R3,F1,F2,F3))
    print("%d\t%d\t%d\t%d\t%d\t%d"% (count_0_cate, count_0_tool, count_0_api,count_0_cate+count_1_cate, \
        count_0_tool+count_1_tool, count_0_api+count_1_api))
    print("%d\t%d\t%d\t%d\t%d\t%d"% (count_2_cate, count_2_tool, count_2_api,count_3_cate+count_2_cate, \
        count_2_tool+count_3_tool, count_2_api+count_3_api))

    #print("%d\t%d\t%d\t%d\t%d\t%d"% (count_cate_match_dfs, count_tool_match_dfs, count_api_match_dfs,count_cate_match_dfsleft,count_tool_match_dfsleft,count_api_match_dfsleft))
    #print('###############Cate|Tool|API|Pass|Total#################')
    #print('%d\t%d\t%d\t%d\t%d\t%d' % (count_cate_match+count_cate_match_dfs, count_tool_match+count_tool_match_dfs, 
    #    count_api_match+count_api_match_dfs,total_case,count_pass,availble))
    #print('%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d' % (count_cate_match+count_cate_match_dfsleft, count_tool_match+count_tool_match_dfsleft, 
    #    count_api_match+count_api_match_dfsleft,total_case,count_cate_match, count_tool_match, count_api_match,total_case,availble))



    def get_size(node):
        size = 1
        if len(node["children"]) == 0:
            node["size"] = size
            return size
        else:
            for child in node["children"]:
                size += get_size(child)
            node["size"] = size
            return size
    def get_leaf_node_count(node):
        thought_count = (1 if node["node_type"] == "Thought" else 0)
        if len(node["children"]) == 0:
            return (1 if node["expand_num"] != 0 else 0), node["Elo"], 1, thought_count
        else:
            result = 0
            max_elo = -1e7
            node_count = 1
            for child in node["children"]:
                child_left_node_count, child_max_elo, child_node_count, child_thought_count = get_leaf_node_count(child)
                result += child_left_node_count
                node_count += child_node_count
                thought_count += child_thought_count
                max_elo = max(max_elo,child_max_elo)
            return result, max_elo, node_count, thought_count

    def recursive_get_error_code(obj):
        result = []
        if type(obj) == dict:
            for key,value in obj.items():
                if key == "observation_code":
                    assert type(value) == int
                    # assert "observation" in obj.keys()
                    if "observation" in obj.keys() and "html" in str(obj["observation"]).lower():
                        result = result + ["html"]
                    else:
                        result = result + [value]

                    # if value == -1:
                    #     print(obj["description"])

                elif key == "description":
                    if "OpenAI service is unavailable" in value:
                        result = result + ["openai"]
                        # print("hello")
                else:
                    # print(f"in {key}")
                    result = result + recursive_get_error_code(value)
        elif type(obj) == list:
            for cont in obj:
                result = result + recursive_get_error_code(cont)
        return result


    def check_real_valid(string):
        fake_true_vocab = ["sorry","apologize","apology","unfortunately","couldn't"]
        for word in fake_true_vocab:
            if word in string.lower():
                return False
        return True



    def classify_N(xs,yss,N):
        zip_value = list(zip(xs,yss[0]))
        zip_value.sort(key = lambda x: x[0])

        threshold =  []
        for i in range(N):
            threshold.append(zip_value[min(((i+1)*len(xs))//(N),len(zip_value)-1)][0])

        bucket = [[] for _ in range(N)]
        for cont in bucket:
            for i in range(len(yss)):
                cont.append([])
        for k,ys in enumerate(yss):
            for x,y in zip(xs,ys):
                for i in range(N):
                    if x < threshold[i]:
                        bucket[i][k].append(y)
                        break
        for i in range(len(bucket)):
            for k in range(len(bucket[i])):
                bucket[i][k] = np.mean(np.array(bucket[i][k]))
        return bucket

    def print_table(table):
        methods = list((table.keys()))
        methods.sort()
        column_names =  ["method"]+list(table[methods[0]].keys())
        for key in table.keys():
            table[key]["method"] = key

        key_length = {}
        for key in column_names:
            if key in ["root/max_Elo","valid_per_data"]:
                continue
            now_max = len(key)
            for method in methods:
                now_max = max(now_max, len(str(table[method][key])))
            key_length[key] = now_max
        
        for key in column_names:
            if key in ["root/max_Elo","valid_per_data"]:
                continue
            # print(key,end=" "*(key_length[key]- len(key))+"|")    

        mode = input_dir[len(input_dir[::-1][input_dir[::-1].find("/"):][::-1]):]
    
        for cnt, method in enumerate(methods):
            for cnt_key, key in enumerate(column_names):
                if key in ["root/max_Elo","valid_per_data"]:
                    continue
                if cnt == 0 and cnt_key == 0:
                    print(mode + "|" + str(table[method][key]),end=" "*(key_length[key]- len(str(table[method][key])))+"|")    
                else:
                    print(str(table[method][key]),end=" "*(key_length[key]- len(str(table[method][key])))+"|")    
            print("")
        for cnt, method in enumerate(methods):
            for cnt_key, key in enumerate(column_names):
                if key in ["root/max_Elo","valid_per_data"]:
                    continue
                print(str(key)+"\t"+str(table[method][key]))  
            print("")

    #print_table(method2result)
