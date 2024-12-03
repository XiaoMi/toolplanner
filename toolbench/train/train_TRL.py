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

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed,create_reference_model
from trl.core import LengthSampler,respond_to_batch

from toolbench.inference.LLM.tool_llama_model import ToolLLaMA
from tqdm import tqdm
from datasets import load_dataset

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



def preprocess(sources,tokenizer: transformers.PreTrainedTokenizer,template: str="tool-llama") -> Dict:
    conv = get_conversation_template(template)
    if template == "tool-llama":
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    elif template == "tool-llama-single-round" or template == "tool-llama-multi-rounds":
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

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
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                continue
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            
            # only train on the last assistant reply, treat the history chat as instruction
            prefix = parts[:-1]
            instruction = ""
            for part in prefix:
                instruction += part
                instruction += sep

            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(instruction).input_ids) - 2

            # Ignore the user instructions
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


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
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
    train_raw_data=train_raw_data[:100]
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)



class LazyRLDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, template="tool-llama"):
        super(LazyRLDataset, self).__init__()
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
        #print(self.raw_data[i].keys())
        messages=[]
        for message in self.raw_data[i]["conversations"]:
            messages.append({"role": message["from"], "content":message["value"]})
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            scores=self.raw_data[i]["reward"],
            conversations=messages,
        )
        self.cached_data_dict[i] = ret

        return ret



def build_dataset(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    rank0_print("Loading data...")
    train_raw_data=[]
    raw_data = json.load(open(data_args.data_path, "r"))
    for i in range(len(raw_data)):
        if "reward" in raw_data[i]:
            train_raw_data.append(raw_data[i])
    eval_raw_data=[]
    raw_data = json.load(open(data_args.eval_data_path, "r"))
    for i in range(len(raw_data)):
        if "reward" in raw_data[i]:
            eval_raw_data.append(raw_data[i])
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    train_dataset = LazyRLDataset(train_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    eval_dataset = LazyRLDataset(eval_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



def train_TRL():
    global local_rank
    #replace_llama_with_condense(ratio=4)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    backbone_model = ToolLLaMA(model_name_or_path=model_args.model_name_or_path)
    
    rank0_print(model_args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    #tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, model_max_length=8192)
    dataset = build_dataset(tokenizer,data_args)

    #data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # get models
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map
    )
    model.config.use_cache = False
    #model = AutoModelForCausalLMWithValueHead.from_pretrained(model_args.model_name_or_path)
    #model_ref = create_reference_model(model)
    # initialize trainer
    ppo_config = PPOConfig(
        batch_size=8,
        model_name=model_args.model_name_or_path,
        learning_rate=5e-5,
        mini_batch_size=1,
        gradient_accumulation_steps=8,
        optimize_cuda_cache=True,
        early_stopping=True,
        target_kl=0.1,
        ppo_epochs=4,
        seed=0,
    )

    # create a ppo trainer
    '''
    ppo_trainer = PPOTrainer(ppo_config, model, 
        ref_model=None, 
        tokenizer=tokenizer,
        dataset=dataset["train_dataset"])
    '''

    data_list=[]
    for epoch,batch in tqdm(enumerate(dataset["train_dataset"])):

        if epoch % 1000==0:
            rank0_print("###################################")
            rank0_print(len(batch["conversations"]))
            rank0_print(batch["scores"])
            rank0_print(batch["input_ids"].size())
            rank0_print(batch["labels"].size())
            rank0_print(batch["attention_mask"].size())
            rank0_print(batch["conversations"][-1])
            rank0_print(batch["conversations"][-2])
            backbone_model.change_messages(batch["conversations"][0:-1])
            #if local_rank == 0:
            #    backbone_model.display_conversation()

            conv = get_conversation_template(data_args.conv_template)
            roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}

            # Apply prompt templates
            conversations = []
            for message in batch["conversations"][0:-1]:
                role = roles[message['role']]
                conv.append_message(role, message['content'])
            conversations.append(conv.get_prompt())


            conversation_history = self.conversation_history
            prompt = ''
            for message in batch["conversations"][0:-1]:
                role = roles[message['role']]
                content = message['content']
                prompt += f"{role}: {content}\n"
            prompt += "Assistant:\n"

            data_list.append({"query":prompt,"target":batch["conversations"][-1]['content'],"reward":batch["scores"]})

            


            output = backbone_model.parse(functions=[],process_id=0) 
            rank0_print(output[0])
            rank0_print(output[3])
            rank0_print("########2#############")

            query_txt = conv.get_prompt()
            query_tensor = tokenizer.encode(query_txt, return_tensors="pt")
            rank0_print(query_tensor.size())
            response_tensor  = respond_to_batch(model, query_tensor)
            rank0_print(response_tensor.size())
            reward = [torch.tensor(batch["scores"])]
            #train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

            #print()
            #rank0_print(conversations[0]) #ends with </s>
            #rank0_print(output[4])#prompt ends with \nAssistant:
    
    '''
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
    '''
if __name__ == "__main__":
    train()