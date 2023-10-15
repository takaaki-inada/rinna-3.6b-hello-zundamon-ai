import subprocess

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

LORA_WEIGHTS='./models/rinna_3b_zundamon_prof_homu_20230615_short/checkpoint-80'

# NOTE: transformers 4.30.0.dev0
subprocess.run(f"cp scripts/adapter_config.json {LORA_WEIGHTS}", shell=True)
subprocess.run(f"cp {LORA_WEIGHTS}/pytorch_model.bin {LORA_WEIGHTS}/adapter_model.bin", shell=True)

def make_prompt(message):
    prompt = f"""ユーザ: {message}
システム: """
#     prompt = f"""### 指示:
# {message}

# ### 回答:
# """
    prompt = prompt.replace('\n', '<NL>')
    return prompt

# モデルの準備
base_model = AutoModelForCausalLM.from_pretrained(
    "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
    load_in_8bit=True,
    device_map="auto",
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)

# LoRAモデルの準備
model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
    # device_map="auto"
)

@torch.no_grad()
def inference_func(prompt, max_new_tokens=128, temperature=1.0, repetition_penalty=1.1, top_p=0.9):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        do_sample=True,
        max_new_tokens=max_new_tokens,
        # temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    return output

prompts = [line.rstrip() for line in open('./datasets/sample_prompts.txt')]
print("instruction, output")
for prompt in prompts:
    instruct_prompt = make_prompt(prompt)
    generated = inference_func(instruct_prompt)
    print(f'"{prompt}", "{generated}"')
