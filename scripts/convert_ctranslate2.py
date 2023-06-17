import os
import subprocess

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_WEIGHTS = "./models/rinna_3b_zundamon_prof_homu_20230615/checkpoint-800"
MERGE_OUTPUT_PATH = "./models/rinna_3b_zundamon_prof_homu_20230615_checkpoint-800"
CT2_OUTPUT_PATH = MERGE_OUTPUT_PATH + "_ct2"


# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
    device_map="auto",
    torch_dtype=torch.float16,
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(
    "rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False
)

# LoRAモデルの準備
peft_model = PeftModel.from_pretrained(model, LORA_WEIGHTS, device_map="auto")

peft_model.merge_and_unload()
if not os.path.exists(MERGE_OUTPUT_PATH):
    os.makedirs(MERGE_OUTPUT_PATH)

model.save_pretrained(MERGE_OUTPUT_PATH)
tokenizer.save_pretrained(MERGE_OUTPUT_PATH)

if not os.path.exists(CT2_OUTPUT_PATH):
    os.makedirs(CT2_OUTPUT_PATH)

subprocess.run(
    [
        "ct2-transformers-converter",
        "--force",
        "--model",
        MERGE_OUTPUT_PATH,
        "--quantization",
        "int8",
        "--output_dir",
        CT2_OUTPUT_PATH,
    ]
)
