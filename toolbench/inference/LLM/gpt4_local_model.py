#!/usr/bin/env python
# coding=utf-8
from typing import Optional, List, Mapping, Any
from termcolor import colored
import json
import random

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import openai
from typing import Optional
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, react_parser
from toolbench.inference.Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT
import requests
import time

class GPT4Local:
    def __init__(self, model="text-davinci-003", openai_key="") -> None:
        super().__init__()
        self.model = model
        self.openai_key = openai_key
        self.chatio = SimpleChatIO()
        self.tokenizer = AutoTokenizer.from_pretrained("ToolLLaMA-7b", use_fast=False, model_max_length=8192)

    def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        max_try = 10
        #print("##############here#############")
        #print("prompt",prompt)
        #print("stop",stop)
        try:
            #input_data = {"prompt": "你好", "history": [], "max_length": 512, "model":"gpt-4-0314"}
            input_data = {"prompt": "你好", "history": [], "max_length": 512, "model":"gpt-4-32k-0314"}
            input_data["prompt"] = prompt
            output = requests.post(url="http://10.221.105.108:19005", json=input_data)
            result = json.loads(output.content)["response"]
            #print("result",result)
            time.sleep(3)
            cycle=0
            while result.startswith("ERROR"):
                time.sleep(10)

                output = requests.post(url="http://10.221.105.108:19005", json=input_data)
                result = json.loads(output.content)["response"]
                if result.startswith("ERROR"):
                    if "Please retry after" in result:
                        timelen=result.split("Please retry after ")[1]
                        timelen=timelen.split(" second.")[0]
                        timelen=int(timelen)
                        time.sleep(timelen+1)
                    else:
                        time.sleep(20)
                    output = requests.post(url="http://10.221.105.108:19005", json=input_data)
                    result = json.loads(output.content)["response"]
                    cycle+=1
                if cycle>5:
                    break
        except Exception as e:
            print(e)
            max_try -= 1
            result="e"
        print("#####################result######################")
        print(result)
        return result

    '''
    input_data = {"prompt": "你好", "history": [], "max_length": 1024, "model":"gpt-4-0314"}
with open("time.jsonl", "w") as f:
    for q in lis:
        input_data["prompt"] = q
        output = requests.post(url="http://10.221.105.108:19004", json=input_data)
        output = json.loads(output.content)["response"]
        print(output)
        dic = {}
        dic["ori_src"] = q
        dic["ori_tgt"] = output
        f.write(json.dumps(dic, ensure_ascii=False) + "\n")
    '''
    '''

                response = openai.Completion.create(
                    engine=self.model,
                    prompt=prompt,
                    temperature=0.5,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop="End Action"
                )
                result = response['choices'][0]['text'].strip()
    '''
        
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(self,functions,process_id,**args):
        self.template="tool-llama-single-round"
        conv = get_conversation_template(self.template)
        if self.template == "tool-llama":
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        elif self.template == "tool-llama-single-round" or self.template == "tool-llama-multi-rounds":
            roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}
        conversation_history = self.conversation_history
        question = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "User":
                question = content
                break
        func_str = ""
        func_list = []
        for function_dict in functions:
            param_str = ""
            api_name = function_dict["name"]
            func_list.append(api_name)
            if "Finish" in api_name:
                param_str = f'"return_type": string, "final_answer": string, '
                api_desc = "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. ALWAYS call this function at the end of your attempt to answer the question finally."
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
            else:
                api_desc = function_dict["description"][function_dict["description"].find("The description of this function is: ")+len("The description of this function is: "):]
                for param_name in function_dict["parameters"]["properties"]:
                    data_type = function_dict["parameters"]["properties"][param_name]["type"]
                    param_str += f'"{param_name}": {data_type}, '
                param_str = "{{" + param_str + "}}"
                func_str += f"{api_name}: {api_desc}. Your input should be a json (args json schema): {param_str} The Action to trigger this API should be {api_name} and the input parameters should be a json dict string. Pay attention to the type of parameters.\n\n"
        func_list = str(func_list)
        prompt = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZEROSHOT.replace("{func_str}", func_str).replace("{func_list}", func_list).replace("{func_list}", func_list).replace("{question}", question)
        prompt = prompt.replace("{{", "{").replace("}}", "}")
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "Assistant":
                prompt += f"\n{content}\n"
            elif role == "Function":
                prompt += f"Observation: {content}\n"
        if functions != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)
        decoded_token_len = len(self.tokenizer(predictions))
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")

        # react format prediction
        thought, action, action_input = react_parser(predictions)
        print("thought",thought)
        print("action",action)
        print("action_input",action_input)
        if "\nEnd Action" in action_input:
            action_input=action_input.split("\nEnd Action")[0]
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": {
                "name": action,
                "arguments": action_input
            }
        }
        return message, 0, decoded_token_len


if __name__ == "__main__":
    llm = Davinci()
    result = llm.prediction("How old are you?")
    print(result)