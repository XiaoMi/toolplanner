import argparse
import json
import os
import random
from enum import auto, Enum
from typing import List, Any, Dict

import numpy as np
import pandas as pd
import re

input_data = "../instruction/G3_query.json"
sample_ids= "../test_query_ids/G3_instruction_test_query_ids.json"


raw_data = json.load(open(input_data, "r"))
output_file="G3_query_100_opendomain_cut_apiStand.json"
output_file1="G3_query_100_cut_api.json"
#output_file2="G3_query_100_ground_cut.json"
#input_data = "multi-tool/train_multi.csv"
#output_file="builddata/cut0901/train_multi.txt"
#output_csv="builddata/cut0901/train_multi.csv"


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

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

count=0
out_list=[]
data=[]
cuttask=[]

test_list1=[]
test_list2=[]
test_list=[]

sample_data = json.load(open(sample_ids, "r"))
ids_list=[]
for ids in sample_data:
    ids_list.append(ids)
print(ids_list)

for index, raw in enumerate(raw_data):
    query_id=raw["query_id"]
    category_list=[]
    tool_list=[]
    api_list1=[]
    if str(query_id) in ids_list:
        print("###############################")
        print(query_id)
        print(raw)
        print(raw.keys())
        query=raw["query"]
        query_id=raw["query_id"]
        api_list=raw["api_list"]
        relevantAPIs=raw["relevant APIs"]
        for api_identity in relevantAPIs:
            tool_name=api_identity[0]
            api_name=api_identity[1]
            for api in api_list:
                if api['tool_name']==tool_name:
                    category_name=api['category_name']
                    break
            tool_name = standardize(tool_name)
            tool_list.append(tool_name)
            api_name = change_name(standardize(api_name))
            api_list1.append(api_name)
            if len(category_name)>0:
                category_name = standardize_category(category_name)
                category_list.append(category_name)


        list1=[]
        for api in api_list:
            newapi={"category_name":api["category_name"],"tool_name":api["tool_name"],"api_name":api["api_name"]}
            list1.append(newapi)

        newquery=query.split(".")[0]+"."
        #newplan=".".join(query.split(".")[1:])
        toolnames=", ".join(api_list1)+"."
        newquery=newquery+"\nAPI: "+toolnames
        test_list.append({"query":newquery,"original_query":query,"query_id":query_id})
        test_list1.append({"query":newquery,"original_query":query,"query_id":query_id,"api_list":list1})
        test_list2.append({"query":newquery,"original_query":query,"query_id":query_id,"relevant APIs":relevantAPIs,"api_list":list1})

json.dump(test_list, open(output_file,"w"), indent=4, ensure_ascii=False)  
