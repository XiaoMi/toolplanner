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
from typing import Dict, Optional,Sequence
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from toolbench.tool_conversation import SeparatorStyle
from toolbench.model.model_adapter import get_conversation_template


import logging
import torch.nn.functional as F
import io

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
    stop_response: bool = field(default=False)    

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
    rrhf_weight: float = field(default=100.0)
    length_penalty: float = field(default=1.0)
    only_use_provide: bool = field(default=False)
    only_use_sample: bool = field(default=False)


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





class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(ScoreDataset, self).__init__()
        logging.warning("Loading data...")
        #with open(data_path, 'r') as f:
        #    lines = f.readlines()
        self.data = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.cached_data_dict = {}
        #self.template = template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    stop_response: bool

    def __call__(self, instances):
        template="tool-llama-single-round"
        conv = get_conversation_template(template)
        roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}


        idxs = []
        all_scores = []
        batch_input_ids = []
        score_mask = []
        labels = []
        for idx, ins in enumerate(instances):
            ins = ins['input_ids'] # hack
            sourcesList = ins['conversations_list']
            scores = ins['scores']
            all_scores.append(scores)
            idxs.append([idx] * len(scores))
            prefix=[]
            for i,sources in enumerate(sourcesList):
                if i ==0:
                    prefix=sources[:-1]
                else:
                    sourcesList[i]=prefix+[sources[-1]]
            #for sources in sourcesList:
            conversations = []
            #for i, source in enumerate(sources):
            for i, source in enumerate(sourcesList):
                conv.messages = []
                #print(source)
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    conv.append_message(role, sentence["value"])
                conversations.append(conv.get_prompt())
            # Tokenize conversations
            input_ids = self.tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            #padding="max_length",
            targets = input_ids.clone()

            # Mask targets. Only compute loss on the assistant outputs.
            sep = conv.sep + conv.roles[-1] + ": "
            for conversation, target in zip(conversations, targets):
                total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
                turns = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_TOKEN_ID
                for i, turn in enumerate(turns):
                    if turn == "":
                        continue
                    turn_len = len(self.tokenizer(turn).input_ids)

                    parts = turn.split(sep)
                    
                    # only train on the last assistant reply, treat the history chat as instruction
                    prefix = parts[:-1]
                    instruction = ""
                    for part in prefix:
                        instruction += part
                        instruction += sep

                    # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                    instruction_len = len(self.tokenizer(instruction).input_ids) - 2

                    # Ignore the user instructions
                    target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                target[cur_len:] = IGNORE_TOKEN_ID

                if True:  # Inspect and check the correctness of masking
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, self.tokenizer.unk_token_id, z)
                    rank0_print(self.tokenizer.decode(z))
                    rank0_print(input_ids.size())
                    rank0_print(cur_len)

                if cur_len < self.tokenizer.model_max_length:
                    if cur_len != total_len:
                        target[:] = IGNORE_TOKEN_ID
                        rank0_print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" (ignored)"
                        )     
                #batch_input_ids.append(input_ids)
                #labels.append(targets)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=IGNORE_TOKEN_ID
        )

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )


def make_supervised_data_module1(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
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
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.conv_template)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, stop_response=data_args.stop_response)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)




class RRHFTrainer(Trainer):
    def gather_logits_labels(self, logits, labels):

        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0 
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
        return scores

    def rrhf_loss(self, scores, idxs, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1) # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1) # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, idxs, rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        if self.args.only_use_provide:
            inputs['input_ids'] = inputs['input_ids'][-2:]
            inputs['attention_mask'] = inputs['attention_mask'][-2:]
            inputs['labels'] = inputs['labels'][-2:]
            inputs["idxs"] = inputs["idxs"][:,-2:]
            inputs["scores"] = inputs["scores"][:,-2:]
        if self.args.only_use_sample:
            inputs['input_ids'] = inputs['input_ids'][:-2]
            inputs['attention_mask'] = inputs['attention_mask'][:-2]
            inputs['labels'] = inputs['labels'][:-2]
            inputs["idxs"] = inputs["idxs"][:,:-2]
            inputs["scores"] = inputs["scores"][:,:-2]
        '''
        logits = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask')) # (batch * cand) * L * V
        rank0_print(inputs.get('input_ids').size())
        rank0_print(inputs.get('attention_mask').size())

        rank0_print(logits)
        rank0_print(logits[0])
        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels"))
        scores = self.get_score(logit_label, inputs.get("labels"))
        rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
        sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss


def train():
    global local_rank

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
    #trainer = Trainer(
    #    model=model, tokenizer=tokenizer, args=training_args, **data_module
    #)
    trainer = RRHFTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
