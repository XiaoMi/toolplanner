import argparse
import json
import os
import random
from enum import auto, Enum
from typing import List, Any, Dict

import numpy as np
import pandas as pd

input_data = "../instruction/G3_query.json"
sample_ids= "../test_query_ids/G3_instruction_test_query_ids.json"


raw_data = json.load(open(input_data, "r"))
output_file="G3_query_100_opendomain.json"
output_file1="G3_query_100.json"
output_file2="G3_query_100_ground.json"
#input_data = "multi-tool/train_multi.csv"
#output_file="builddata/cut0901/train_multi.txt"
#output_csv="builddata/cut0901/train_multi.csv"


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
    if str(query_id) in ids_list:
        print(query_id)
        query=raw["query"]
        query_id=raw["query_id"]
        api_list=raw["api_list"]
        relevantAPIs=raw["relevant APIs"]
        list1=[]
        for api in api_list:
            newapi={"category_name":api["category_name"],"tool_name":api["tool_name"],"api_name":api["api_name"]}
            list1.append(newapi)
        test_list.append({"query":query,"query_id":query_id})
        test_list1.append({"query":query,"query_id":query_id,"api_list":list1})
        test_list2.append({"query":query,"query_id":query_id,"relevant APIs":relevantAPIs,"api_list":list1})

json.dump(test_list, open(output_file,"w"), indent=4, ensure_ascii=False)  
json.dump(test_list1, open(output_file1,"w"), indent=4, ensure_ascii=False)  
json.dump(test_list2, open(output_file2,"w"), indent=4, ensure_ascii=False)  
