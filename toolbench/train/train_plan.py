# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from toolbench.tool_conversation import SeparatorStyle
from toolbench.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
torch.set_printoptions(profile="full")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    lazy_preprocess: bool = False
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str="tool-llama"
) -> Dict:
    #print("template",template)#template tool-llama-single-round
    #tool-llama-plan-round make model train 2th sen and last sen
    template="tool-llama-plan-round"
    conv = get_conversation_template(template)
    if template == "tool-llama":
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds" or template=="tool-llama-plan-round":
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}
    #print("roles",roles)#roles {'system': 'System', 'user': 'User', 'function': 'Function', 'assistant': 'Assistant'}
    # Apply prompt templates
    roleList=[]
    conversations = []
    #print("soruce len",len(sources))
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
            roleList.append(role)
        roleList.append("#################")
        conversations.append(conv.get_prompt())
        #print("role","conv")
        #print(source)
        #print(conv.get_prompt())
    
    #print("conversations",len(conversations))
    #['System', 'Assistant', 'System', 'User', 'Assistant', 'Function', 'User', 'Assistant', 'Function', 'Assistant', 'Function', 'Assistant', '########

    #print("conversations")
    #print(len(conversations))##1
    #for index in range(len(conversations)):
    #    print("###############")
    #    print(index)
    #    print(conversations[index])

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[-1] + ": "

    #print("targets")
    #print(targets)
    #print("sep")
    #print(sep)#\nAssistant:
    #print("conv.sep")
    #print(conv.sep)#\n
    #print("conv.sep2")
    #print(conv.sep2)#</s>
    #print("IGNORE_TOKEN_ID")
    #print(IGNORE_TOKEN_ID)#-100
    #[step1 sep1 step2 sep1 step3 ,,, stepn-1 sep2 stepn]
    #print("conversations len:",len(conversations))
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        #print(total_len)#3613
        #print("turns_len",len(turns))#2
        #print("**********************************************")
        for i, turn in enumerate(turns):
            #print("i")
            #print("i",i)
            #print("turn")
            #print("turn",len(turn))
            if turn == "":
                continue
            turn_len = len(tokenizer(turn).input_ids)
            #print(turn_len)#2212

            parts = turn.split(sep)

            
            # only train on the last assistant reply, treat the history chat as instruction
            prefix = parts[:-1]
            instruction = ""
            for part in prefix:
                instruction += part
                instruction += sep

            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(instruction).input_ids) - 2

            #print(parts)# 1. planPrompt  2. plan+taskPrompt 3. func
            #1. System  2. Assistant+System+User 3. Assistant
            #1. System 2. Assistant+System+User 3. Assistant+Function 4. Assistant
            #print("roleList",roleList)
            #print("parts",len(parts))#3
            #['System', 'Assistant', 'System', 'User', 'Assistant', '#################']
            #['System', 'Assistant', 'System', 'User', 'Assistant', 'Function', 'Assistant', '#################']
            #['System', 'Assistant', 'System', 'User', 'Assistant', 'Function', 'Assistant', 'Function', 'Assistant', '#################']
            #for index in range(len(parts)):
            #    print("####################################")
            #    print(index)
            #    print(parts[index])

            #print("instruction")
            #print(instruction)
            #print("cur_len")
            #print(cur_len)
            #print(instruction_len)

            # Ignore the user instructions
            #[1:2167]
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    #print("##########################end#############################")
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template="tool-llama"):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        self.template = template
        data_dict = preprocess(sources, tokenizer, self.template)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template="tool-llama"):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.template = template

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.template)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    print("############################")
    print(data_args.lazy_preprocess)#True
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))
    if data_args.eval_data_path is not None:
        train_raw_data = raw_data
        eval_raw_data = json.load(open(data_args.eval_data_path, "r"))
    else:
        # Split train/test
        perm = np.random.permutation(len(raw_data))
        split = int(len(perm) * 0.98)
        train_indices = perm[:split]
        eval_indices = perm[split:]
        train_raw_data = [raw_data[i] for i in train_indices]
        eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    print("train_raw_data")
    print(train_raw_data[0])
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    print("#####################train_cut#########################")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map
    )
    model.config.use_cache = False
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
