# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
import transformers
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from torch.utils.data import DataLoader

from toolbench.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from toolbench.train.llama_condense_monkey_patch import (
    replace_llama_with_condense,
)
# ratio = 4 means the sequence length is expanded by 4, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
replace_llama_with_condense(ratio=4)

tqdm.pandas()


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )

    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    data_path:Optional[str] = field(default="", metadata={"help": "the model name"})
    eval_data_path:Optional[str] = field(default="", metadata={"help": "the model name"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name
dataset_name = script_args.data_path
config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
)

output_data_path="data/category/dataset/G3_plan_gen_train_1026_G3_3tag_whole_TraceAll_Reward.json"    
train_dataset = load_dataset("json", data_files=output_data_path, split="train")
#train_dataset = train_dataset.select(range(100000))

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

#tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.

#if getattr(tokenizer, "pad_token", None) is None:
#    tokenizer.pad_token = tokenizer.eos_token
tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.model_name,
    model_max_length=8192,
    padding_side="right",
    use_fast=False,
)
#tokenizer.pad_token = tokenizer.unk_token
if tokenizer.pad_token_id == None:
    tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    tokenizer,
    dataset_name="lvwerra/stack-exchange-paired",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    # load imdb with datasets
    #ds = load_dataset(dataset_name, data_dir="data/rl", split="train")
    output_data_path="data/category/dataset/G3_plan_gen_train_1026_G3_3tag_whole_TraceAll_Reward.json"
    #ds=load_dataset("json", data_files=dataset_name, split="train")
    #original_columns = ds.column_names
    num_proc = 24


    def preprocess_function(examples):
        #batch_size=len(examples["query"])
        #print(batch_size)
        #print(examples["target"])
        new_examples = {
            "query": [],
            "input_ids": [],
            "target":[],
            "target_ids":[],
            "reward":[],
        }
        for query in  examples["query"]:
            input_ids = tokenizer(query,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            new_examples["query"].append(query)
            new_examples["input_ids"].append(input_ids[0])

        for target in examples["target"]:
            target_ids = tokenizer(target,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            new_examples["target"].append(target)
            new_examples["target_ids"].append(target_ids[0])

        for reward in examples["reward"]:
            new_examples["reward"].append(reward)

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        
    )
    #remove_columns=original_columns,
    #ds = ds.filter(lambda x: len(x["input_ids"]) < 8192, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer,script_args.data_path)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
'''
def collator(data):
    return dict(
            query=[d["query"] for d in data],
            input_ids=[d["input_ids"] for d in data],
            target=[d["target"] for d in data],
            target_ids=[d["target_ids"] for d in data],
            reward=[d["target_ids"] for d in data],
        )
'''
# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
'''
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)

'''


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

train_dataloader= DataLoader(dataset=dataset, batch_size=script_args.batch_size, shuffle=True,num_workers=4)
#for epoch, batch in tqdm(enumerate(dataset)):
for epoch, batch in tqdm(enumerate(train_dataloader)):
#for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if epoch%200==0:
        print(len(batch["input_ids"]))
        print(batch.keys())
        print(len(batch["target_ids"]))
        print(batch["input_ids"].size())
        print(batch["target_ids"].size())
        print(batch["reward"])
        print(batch["reward"].dtype)
        #print(batch["rewards"].size())
        
        print(config.total_ppo_epochs)
        print(script_args.save_freq)
    if epoch >= config.total_ppo_epochs:
        break

    question_tensors = [d for d in batch["input_ids"]]

    '''
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    '''
    response_tensors=[d for d in batch["target_ids"]]
    #batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score (using the sentiment analysis pipeline)
    #texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    #pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    #rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
    rewards=[torch.tensor(d) for d in batch["reward"]]
    #rewards=torch.tensor(batch["reward"])
    # Run PPO step
    #stats = ppo_trainer.step(question_tensors,response_tensors, rewards)
    stats = ppo_trainer.step(question_tensors,response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
