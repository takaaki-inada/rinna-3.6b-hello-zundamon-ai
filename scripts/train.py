import os

import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import transformers
from peft import (LoraConfig, TaskType, get_peft_model,
                  get_peft_model_state_dict, prepare_model_for_int8_training,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from transformers.trainer_callback import TrainerCallback

from datasets import Dataset, load_dataset

OUTPUT = "./models/rinna_3b_zundamon_prof_homu_20230615/"

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 32
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 100
# LoRAのLEARNING_RATEは 5e-4, 3e-4 が良さそう
# LEARNING_RATE = 5e-4
LEARNING_RATE = 3e-4
CUTOFF_LEN = 256
# LORA_R = 16 にすると RTX 3060 で以下のエラーが出る
# Error invalid device ordinal at line 359 in file /home/tim/git/bitsandbytes/csrc/pythonInterface.c /arrow/cpp/src/arrow/filesystem/s3fs.cc:2598: arrow::fs::FinalizeS3 was not called even though S3 was initialized.
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0.1
VAL_SET_SIZE_HOMU = 0.1
TARGET_MODULES = [
    "query_key_value"
]

DATA_PATH = './datasets/databricks-dolly-15k-ja-zundamon.json'

DATA_ZUNDA_PROF_PATH = './datasets/zundamon_prof_homu_20230615.csv'

device_map = "auto"
world_size = int(os.environ.get('WORLD_SIZE', 1))
ddp = world_size != 1
if ddp:
    device_map = {'':int(os.environ.get('LOCAL_RANK') or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map=device_map,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

data = load_dataset("json", data_files=DATA_PATH)
print(f"データセットの件数: {data.num_rows}")
data_zunda_prof = load_dataset("csv", data_files=DATA_ZUNDA_PROF_PATH)
print(f"データセットの件数: {data_zunda_prof.num_rows}")

train_val = data_zunda_prof["train"].train_test_split(
    test_size=VAL_SET_SIZE_HOMU, shuffle=True, seed=42
)
train_data_homu = train_val["train"]
val_data_homu = train_val["test"]

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = pd.DataFrame(train_val["train"].filter(lambda x: x["category"] == 'general_qa'))
val_data = pd.DataFrame(train_val["test"].filter(lambda x: x["category"] == 'general_qa'))

train_data = Dataset.from_pandas(pd.concat([train_data, pd.DataFrame(train_data_homu)]))
val_data = Dataset.from_pandas(pd.concat([val_data, pd.DataFrame(val_data_homu)]))


def generate_prompt(data_point):
    # if data_point["input"]:
    if False:
        result = f"""### 指示:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 回答:
{data_point["output"]}"""
    else:
        result = f"""ユーザ: {data_point["instruction"]}
システム: {data_point["output"]}"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result


def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
# print(len(train_data), len(val_data))

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        output_dir=OUTPUT,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        save_total_limit=None,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        optim="paged_adamw_8bit",
        # auto_find_batch_size=True,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.config.use_cache = True

model.save_pretrained(f"{OUTPUT}last")
